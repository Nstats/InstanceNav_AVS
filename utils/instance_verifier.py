"""Verify described goal instances from RGB and candidate boxes, without oracle IDs."""
import json
import logging
import hashlib
import os
import re
import csv
import sys
import time
from functools import lru_cache
from pathlib import Path
from uuid import uuid4
from PIL import Image, ImageDraw
from configs.navigation_config import (instance_verification_prompt, instance_verification_scope,
                                       instance_rejection_policy, env_bool)
from utils.model_client import ModelRequestError

LOG = logging.getLogger(__name__)
# INSTANCE_VERIFICATION_PROMPT selects the rule sentence; the "yes" wording is identical.
# match is the original text, byte for byte, so its requests and cache keys are unchanged.
POSITIVE_MATCH_RULES = {
    'match': 'If evidence is inadequate, answer unknown. ',
    'match_strict_no': (
        'Answer "no" only if the visible evidence clearly contradicts the description, so that this '
        'is definitely not the described object. If the object is partly hidden, too small or '
        'unclear, if the described attributes or surrounding context are not visible in this view, '
        'or if you cannot tell for any other reason, answer "unknown", not "no". '),
}
DEFAULT_FACTOR_FILE = Path(__file__).resolve().parents[1] / 'data/instancenav_factors/val10_factors_v1_review.csv'
FACTOR_FIELDS = ('factor_id', 'kind', 'attribute', 'value', 'reference_category',
                 'reference_attributes', 'source_span')
FACTOR_STATUSES = ('not_contradicted', 'contradicted', 'unknown')


@lru_cache(maxsize=4)
def _load_factor_index(path):
    index = {}
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'episode_key', 'category', *FACTOR_FIELDS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Invalid InstanceNav factor CSV columns: {path}')
        for row in reader:
            if any(row.get(field) is None for field in required):
                raise ValueError(f'Incomplete factor CSV row: {reader.line_num}')
            key = row['episode_key'].strip()
            factor = {field: row[field].strip() for field in FACTOR_FIELDS}
            if (not key or not row['category'].strip() or not factor['factor_id']
                    or not factor['source_span'] or not factor['value']
                    or factor['kind'] not in ('attribute', 'relation')):
                raise ValueError(f'Invalid factor CSV row: {reader.line_num}')
            category, factors = index.setdefault(key, (row['category'].strip(), []))
            if category != row['category'].strip() or any(
                    f['factor_id'] == factor['factor_id'] for f in factors):
                raise ValueError(f'Duplicate factor or inconsistent category for {key}')
            factors.append(factor)
    return index


def constraints_for_episode(episode, path=DEFAULT_FACTOR_FILE):
    """Look up task text by stable dataset key, never by runtime episode order."""
    key = f'{episode.goals_key}__ep{episode.episode_id}'
    entry = _load_factor_index(str(Path(path).resolve())).get(key)
    if entry is None:
        raise ValueError(f'No subconstraints for InstanceNav task {key} in {path}')
    category, factors = entry
    if category != episode.object_category:
        raise ValueError(f'Factor category mismatch for {key}: {category}')
    # Return copies: episode state must not mutate the process-wide CSV cache.
    return [dict(factor) for factor in factors]


def _candidate_image(rgb, boxes):
    image = Image.fromarray(rgb[..., :3]).convert('RGB')
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(x) for x in box]
        draw.rectangle((x1, y1, x2, y2), outline='red', width=3)
        draw.text((x1+3, y1+3), str(i), fill='red', stroke_width=1, stroke_fill='white')
    return image


def parse_matches(answer, count):
    answer = re.sub(r'^```(?:json)?\s*|\s*```$', '', answer.strip(), flags=re.IGNORECASE)
    data = json.loads(answer)
    # Qwen may wrap the requested object in a singleton list even with a
    # schema. Unwrap only this unambiguous form; never merge candidate lists,
    # pad missing decisions, or infer decisions from explanatory text.
    if (isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict)
            and set(data[0]) == {'matches'}):
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError('expected a JSON object')
    matches = data['matches']
    if not isinstance(matches, list) or len(matches) != count:
        raise ValueError('wrong matches type or length')
    if any(not isinstance(value, str) for value in matches):
        raise ValueError('non-string decision')
    matches = [value.strip().lower() for value in matches]
    if any(value not in ('yes', 'no', 'unknown') for value in matches):
        raise ValueError('unrecognized decision')
    return matches


def record_invalid_verification(prompt, image, schema, answer, error):
    """Keep a bounded replay sample, without credentials or encoded API images."""
    LOG.warning('Invalid instance verification: %s: %s', type(error).__name__, error)
    directory = os.environ.get('PSGNAV_API_DIAGNOSTICS_DIR')
    if not directory:
        return
    try:
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        if len(list(folder.glob('instance_*.json'))) >= 20:
            return
        stem = 'instance_' + uuid4().hex
        image_name = stem + '.png'
        image.save(folder / image_name)
        data = {'prompt': prompt, 'schema': schema, 'answer': answer[:20000],
                'error': f'{type(error).__name__}: {error}', 'image': image_name,
                'image_sha256': hashlib.sha256(image.tobytes()).hexdigest()}
        (folder / (stem + '.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except OSError as exc:
        LOG.warning('Cannot save verification diagnostic: %s', type(exc).__name__)


def log_instance_verification(event, *, context=None, **fields):
    """Write one JSON record to stdout, captured by run_instance100.sh's log."""
    record = {**(context or {}), 'event': event, **fields}
    if record.get('collect_directory'):
        from utils.collection_recorder import append_record
        append_record(record['collect_directory'], 'Instance Verification', record)
    print('[Instance Verification] ' + json.dumps(record, ensure_ascii=True), flush=True)


def _complete_logged_verification(model_client, request, image, response_format, parse, context):
    """Log before validation so malformed replies and cached replies are visible."""
    validated_answer = object()
    parsed = None

    def validate(answer):
        nonlocal validated_answer, parsed
        metadata = {}
        if context.get('collect_directory'):
            from utils.collection_recorder import request_metadata
            metadata = request_metadata(model_client)
        log_instance_verification('raw_response', context=context, answer=answer, **metadata)
        try:
            parsed = parse(answer)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            log_instance_verification('format_error', context=context,
                                      error=f'{type(error).__name__}: {error}')
            record_invalid_verification(request, image, response_format['json_schema']['schema'], answer, error)
            raise
        validated_answer = answer

    try:
        answer = model_client.complete(request, image, response_format=response_format, validator=validate)
    except ModelRequestError as error:
        log_instance_verification('request_error', context=context,
                                  error=f'{type(error).__name__}: {error}')
        raise
    # Some compatible clients return the answer without invoking the validator.
    # Validate/log that answer too, but never duplicate a callback's valid record.
    if answer != validated_answer:
        validate(answer)
    return parsed


def verify_candidates(model_client, rgb, boxes, category, attributes, *, conflict=False, log_context=None):
    """Compatibility API: preserve the original kept-index result and request behavior."""
    return _verify_candidate_result(model_client, rgb, boxes, category, attributes,
                                    conflict=conflict, log_context=log_context)['kept']


def verify_candidate_decisions(model_client, rgb, boxes, category, attributes, *, log_context=None):
    """Full positive-match decisions, without conflating no and unknown or requesting twice."""
    return _verify_candidate_result(model_client, rgb, boxes, category, attributes,
                                    log_context=log_context)['matches']


def _verify_candidate_result(model_client, rgb, boxes, category, attributes, *, conflict=False, log_context=None):
    if not boxes:
        return {'kept': [], 'matches': []}
    # Only language descriptions are sent; annotation image paths and object IDs are excluded.
    description = '\n'.join(str(attributes.get(k, '')) for k in
                            ('intrinsic_attributes', 'extrinsic_attributes')).strip()
    scope = instance_verification_scope()
    if scope == 'category' and conflict:
        raise ValueError('Category verification requires positive_match')
    if scope == 'instance' and not description:
        raise ValueError('InstanceNav requires intrinsic/extrinsic goal descriptions')
    image = _candidate_image(rgb, boxes)
    if conflict:
        judgment = (
            'For each box, decide whether the object has a significant conflict with these intrinsic and extrinsic attributes. '
            'Answer "yes" for a significant conflict, "no" when no significant conflict is visible, '
            'and "unknown" when the evidence is inadequate to establish a conflict. '
        )
    else:
        judgment = (
            'For each box, decide whether its object matches the intrinsic attributes and visible surrounding context. '
            'Matching the object category alone is not sufficient. '
            + POSITIVE_MATCH_RULES[instance_verification_prompt()]
        )
    prompt = (
        f'Find the specific {category} described below. Red boxes are candidate objects numbered 0 to {len(boxes)-1}.\n'
        f'Target description:\n{description}\n'
        f'{judgment}'
        f'Return only one JSON object with a "matches" array of exactly {len(boxes)} strings. '
        'Each string must be "yes", "no", or "unknown", one decision per box in index order. '
        'No additional fields, ellipsis, comments, or explanations.'
    )
    if scope == 'category':
        # No instance description is included; keep image, schema and stage unchanged.
        prompt = (
            f'Check object category only. Red boxes are candidate objects numbered 0 to {len(boxes)-1}.\n'
            f'Target category: {category}.\n'
            'For each box, answer "yes" when the visible evidence supports that the boxed object '
            'belongs to the target category. Answer "no" only when visible evidence clearly shows '
            'that it does not belong to the target category. If the object is too small, obscured, '
            'unclear, or you cannot determine its category, answer "unknown". '
            'Do not distinguish between individual instances of the same category. '
            f'Return only one JSON object with a "matches" array of exactly {len(boxes)} strings. '
            'Each string must be "yes", "no", or "unknown", one decision per box in index order. '
            'No additional fields, ellipsis, comments, or explanations.'
        )
    # The admission check is independent of confirmation counting, including
    # when the image and boxes are identical. Distinct prompts also separate
    # their keys in ModelClient's exact-response cache.
    stage = (log_context or {}).get('stage')
    if stage == 'candidate_admission':
        prompt = 'Stage: candidate admission. This check grants approach eligibility only; it is not a goal-confirmation vote.\n' + prompt
    elif stage == 'goal_confirmation':
        prompt = 'Stage: goal confirmation. Independently evaluate this admitted candidate; do not assume any earlier admission check passed.\n' + prompt
    schema = {
        'type': 'object',
        'properties': {'matches': {'type': 'array',
                                   'items': {'type': 'string', 'enum': ['yes', 'no', 'unknown']},
                                   'minItems': len(boxes), 'maxItems': len(boxes)}},
        'required': ['matches'], 'additionalProperties': False,
    }
    response_format = {'type': 'json_schema', 'json_schema': {
        'name': 'instance_matches', 'strict': True, 'schema': schema}}
    log_context = {'verification_id': uuid4().hex, **(log_context or {}),
                   'mode': 'contradiction' if conflict else 'instance_match',
                   'category': category, 'candidate_count': len(boxes),
                   'verification_scope': scope, 'rejection_policy': instance_rejection_policy(),
                   'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
    if env_bool('INSTANCE_VERIFICATION_RECORD_INPUTS', False) and not log_context.get('collect_directory'):
        logfile = os.environ.get('PSGNAV_LOG_FILE')
        if not logfile:
            raise ValueError('Input recording requires PSGNAV_LOG_FILE')
        directory = Path(logfile).resolve().with_suffix('.verification')
        directory.mkdir(parents=True, exist_ok=True)
        log_context['collect_directory'] = str(directory)
    for attempt in range(3):
        request = prompt
        if attempt:
            # Change the formatting instruction so a malformed cached reply is
            # never replayed as the retry. The target and visual evidence stay identical.
            request += (f'\nFormatting correction attempt {attempt}: return one JSON object '
                        f'with a "matches" array of exactly {len(boxes)} strings. '
                        'Each string must be "yes", "no", or "unknown". '
                        'No ellipsis, comments, explanation, or extra entries.')
        context = {**log_context, 'attempt': attempt + 1}
        if context.get('collect_directory'):
            from utils.collection_recorder import save_image, model_settings
            marked = save_image(context['collect_directory'], image)
            log_instance_verification('request', context=context, prompt=request,
                                      response_format=response_format, image=marked,
                                      model_settings=model_settings(model_client))
        try:
            matches = _complete_logged_verification(
                model_client, request, image, response_format,
                lambda answer: parse_matches(answer, len(boxes)), context)
        except (ValueError, TypeError, KeyError, AttributeError):
            LOG.warning('Instance verification malformed response: boxes=%d attempt=%d; retrying format only',
                        len(boxes), attempt + 1)
            continue
        if conflict:
            kept = [i for i, value in enumerate(matches) if value != 'yes']
            reported = log_context.get('box_indices', list(range(len(matches))))
            for box_index, decision in zip(reported, matches):
                print(f'[Contradiction Check] step={log_context.get("step")} '
                      f'box={box_index} decision={decision}', flush=True)
        else:
            kept = [i for i, value in enumerate(matches) if value == 'yes']
        log_instance_verification('result', context=context, matches=matches, kept_candidate_ids=kept)
        return {'kept': kept, 'matches': matches}
    log_instance_verification('failed', context=log_context, attempts=3, reason='invalid_response_format')
    raise ModelRequestError('Invalid InstanceNav verification JSON after 3 format attempts') from None


def _unique_json_object(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f'Duplicate JSON field: {key}')
        data[key] = value
    return data


def parse_post_confirmation_check(answer, image_count):
    """Accept only a complete, unambiguous decision with a valid evidence index."""
    answer = re.sub(r'^```(?:json)?\s*|\s*```$', '', answer.strip(), flags=re.IGNORECASE)
    data = json.loads(answer, object_pairs_hook=_unique_json_object)
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if not isinstance(data, dict) or set(data) != {'decision', 'image_index', 'reason'}:
        raise ValueError('expected exactly decision, image_index, reason')
    if data['decision'] not in ('conflict', 'no_conflict', 'unknown'):
        raise ValueError('decision must be conflict, no_conflict, or unknown')
    if type(data['image_index']) is not int or not -1 <= data['image_index'] < image_count:
        raise ValueError('image_index is outside the supplied image set')
    if not isinstance(data['reason'], str) or len(data['reason']) > 240:
        raise ValueError('reason must be a string of at most 240 characters')
    if data['decision'] == 'conflict':
        if data['image_index'] < 0 or not data['reason'].strip():
            raise ValueError('a conflict requires an image_index and a concrete reason')
    elif data['image_index'] != -1:
        raise ValueError('non-conflict decisions require image_index=-1')
    return data


def _save_post_confirmation_inputs(images, prompt, context):
    """Save the exact multi-view input next to an evaluation log for replay."""
    logfile = os.environ.get('PSGNAV_LOG_FILE')
    if not logfile:
        return None
    folder = Path(logfile).with_suffix('.rechecks') / context['verification_id']
    try:
        folder.mkdir(parents=True, exist_ok=True)
        names = []
        for index, image in enumerate(images):
            name = f'image_{index:04d}.png'
            image.save(folder / name)
            names.append(name)
        (folder / 'input.json').write_text(json.dumps(
            {'context': context, 'prompt': prompt, 'images': names}, ensure_ascii=False, indent=2))
        return str(folder)
    except OSError as error:
        LOG.warning('Cannot save post-confirmation evidence: %s', error)
        return None


def verify_post_confirmation(model_client, images, category, attributes, *, log_context=None):
    """One multi-image veto. Exactly three total API/format attempts, never 3x3."""
    if not images:
        raise ValueError('Post-confirmation checking requires candidate observations')
    description = '\n'.join(str(attributes.get(k, '')) for k in
                            ('intrinsic_attributes', 'extrinsic_attributes')).strip()
    if not description:
        raise ValueError('InstanceNav requires intrinsic/extrinsic goal descriptions')
    prompt = (
        f'Inspect the red-boxed {category} across all {len(images)} numbered images. '
        'The boxes track one candidate across views. Judge only that object. '
        'The last image is the current observation; the others are its earlier observations.\n'
        f'Target description:\n{description}\n'
        'Look for clear, significant visual evidence that conflicts with the target description. '
        'Do not require every attribute or surrounding object to be visible. '
        'Occlusion, a missing reference object, uncertain material, lighting changes, minor color '
        'differences, and viewpoint-dependent left/right differences are not significant conflicts. '
        'Do not assume the candidate is correct or incorrect based on how the images were selected. '
        'Use conflict only for a concrete visible incompatibility. Use no_conflict when none is '
        'visible, and unknown when the evidence is insufficient to establish a conflict.\n'
        'Return ONE short JSON object with exactly three fields: decision, image_index, reason. '
        'decision must be conflict, no_conflict, or unknown. For conflict, image_index must identify '
        'one supplied image with the clearest evidence, and reason must briefly name the target '
        'attribute and the incompatible observation (at most 240 characters). Otherwise use '
        'image_index=-1 and reason="". No per-image list, markdown, or text outside the JSON.\n'
        'Example with no conflict: {"decision":"no_conflict","image_index":-1,"reason":""}\n'
        'Example with insufficient evidence: {"decision":"unknown","image_index":-1,"reason":""}'
    )
    schema = {
        'type': 'object', 'properties': {
            'decision': {'type': 'string', 'enum': ['conflict', 'no_conflict', 'unknown']},
            'image_index': {'type': 'integer', 'minimum': -1, 'maximum': len(images) - 1},
            'reason': {'type': 'string', 'maxLength': 240},
        }, 'required': ['decision', 'image_index', 'reason'], 'additionalProperties': False,
    }
    response_format = {'type': 'json_schema', 'json_schema': {
        'name': 'post_confirmation_contradiction', 'strict': True, 'schema': schema}}
    context = {**(log_context or {}), 'verification_id': uuid4().hex,
               'mode': 'post_confirmation_contradiction', 'category': category,
               'image_count': len(images),
               'image_hashes': [hashlib.sha256(im.tobytes()).hexdigest() for im in images]}
    evidence_dir = _save_post_confirmation_inputs(images, prompt, context)
    log_instance_verification('request', context=context, evidence_dir=evidence_dir)
    failures = []
    for attempt in range(1, 4):
        attempt_context = {**context, 'attempt': attempt}
        parsed = None
        validated = object()

        def validate(answer):
            nonlocal parsed, validated
            log_instance_verification('raw_response', context=attempt_context, answer=answer)
            parsed = parse_post_confirmation_check(answer, len(images))
            validated = answer

        request = prompt
        if attempt > 1:
            request += (f'\nResponse attempt {attempt}: keep exactly the same visual judgment task. '
                        'Return only the complete three-field JSON object specified above.')
        try:
            # The outer loop owns the entire three-attempt budget, including HTTP failures.
            answer = model_client.complete(request, images=images, response_format=response_format,
                                           validator=validate, max_tokens=512, retries=0)
            if answer != validated:
                validate(answer)
        except (ModelRequestError, ValueError, TypeError, KeyError, AttributeError) as error:
            reason = f'{type(error).__name__}: {error}'
            failures.append(f'attempt {attempt}: {reason}')
            log_instance_verification('request_error' if isinstance(error, ModelRequestError)
                                      else 'format_error', context=attempt_context, error=reason)
            if attempt < 3 and isinstance(error, ModelRequestError):
                time.sleep(attempt)
            continue
        passed = parsed['decision'] != 'conflict'
        log_instance_verification('result', context=attempt_context, passed=passed, **parsed)
        return parsed
    message = 'Post-confirmation contradiction check failed after 3 attempts; ' + '; '.join(failures)
    log_instance_verification('failed', context=context, attempts=3, errors=failures)
    print(message, file=sys.stderr, flush=True)
    raise ModelRequestError(message) from None


def parse_constraint_results(answer, count, factor_ids):
    """Validate the complete result before any candidate or history is updated."""
    answer = re.sub(r'^```(?:json)?\s*|\s*```$', '', answer.strip(), flags=re.IGNORECASE)
    data = json.loads(answer, object_pairs_hook=_unique_json_object)
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict) or set(data) != {'candidates'}:
        raise ValueError('expected only a candidates object')
    candidates = data['candidates']
    if not isinstance(candidates, list) or len(candidates) != count:
        raise ValueError('wrong candidate count')
    for i, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != {'candidate_id', 'constraints'}:
            raise ValueError('invalid candidate fields')
        if type(candidate['candidate_id']) is not int or candidate['candidate_id'] != i:
            raise ValueError('wrong candidate ID or order')
        results = candidate['constraints']
        if not isinstance(results, list) or len(results) != len(factor_ids):
            raise ValueError('wrong factor count')
        for factor_id, result in zip(factor_ids, results):
            if not isinstance(result, dict) or set(result) != {'factor_id', 'status', 'evidence'}:
                raise ValueError('invalid factor fields')
            if result['factor_id'] != factor_id:
                raise ValueError('wrong factor ID or order')
            if result['status'] not in FACTOR_STATUSES:
                raise ValueError('unrecognized factor status')
            if not isinstance(result['evidence'], str) or not result['evidence'].strip():
                raise ValueError('missing factor evidence')
    return candidates


def _constraint_schema(count, factor_ids):
    factor = {
        'type': 'object',
        'properties': {
            'factor_id': {'type': 'string', 'enum': factor_ids},
            'status': {'type': 'string', 'enum': list(FACTOR_STATUSES)},
            'evidence': {'type': 'string'},
        },
        'required': ['factor_id', 'status', 'evidence'], 'additionalProperties': False,
    }
    candidate = {
        'type': 'object',
        'properties': {
            'candidate_id': {'type': 'integer', 'enum': list(range(count))},
            'constraints': {'type': 'array', 'items': factor,
                            'minItems': len(factor_ids), 'maxItems': len(factor_ids)},
        },
        'required': ['candidate_id', 'constraints'], 'additionalProperties': False,
    }
    return {
        'type': 'object',
        'properties': {'candidates': {'type': 'array', 'items': candidate,
                                      'minItems': count, 'maxItems': count}},
        'required': ['candidates'], 'additionalProperties': False,
    }


def build_constraint_prompt(category, attributes, constraints, count):
    description = '\n'.join(str((attributes or {}).get(k, '')) for k in
                            ('intrinsic_attributes', 'extrinsic_attributes')).strip()
    if not description:
        raise ValueError('InstanceNav requires intrinsic/extrinsic goal descriptions')
    factors = [{field: factor[field] for field in FACTOR_FIELDS} for factor in constraints]
    factor_ids = [factor['factor_id'] for factor in factors]
    if not factor_ids or any(not isinstance(f, str) or not f for f in factor_ids) or len(set(factor_ids)) != len(factor_ids):
        raise ValueError('Expected nonempty, unique subconstraint IDs')

    # Match the current array sizes and IDs; the example is not a visual judgment.
    example_evidence = (
        'In this fictional example, the visible feature is compatible with the constraint.',
        'In this fictional example, a clearly visible feature conflicts with the stated requirement.',
        'In this fictional example, occlusion prevents checking this constraint.',
    )
    examples = []
    # With only one or two result slots, provide small alternative examples so
    # all three states are illustrated without adding fictitious factor IDs.
    for offset in range(max(1, 3 - count * len(factor_ids) + 1)):
        example = {'candidates': [
            {'candidate_id': i, 'constraints': [
                {'factor_id': factor_id,
                 'status': FACTOR_STATUSES[(i * len(factor_ids) + j + offset) % 3],
                 'evidence': example_evidence[(i * len(factor_ids) + j + offset) % 3]}
                for j, factor_id in enumerate(factor_ids)]}
            for i in range(count)]}
        examples.append(json.dumps(example, ensure_ascii=True, indent=2))
    return (
        'You inspect candidate objects for clear, significant visual contradictions '
        "with the navigation target's constraints.\n"
        'The RGB image contains red boxes from an upstream visual detector. '
        'Each number identifies a candidate. Inspect that object; use the surrounding '
        "image for relations. Do not attribute another object's features to this candidate.\n"
        f'Target category: {category}\nTarget description:\n{description}\n'
        f'Candidate IDs for this request: {list(range(count))}\n'
        'Check every constraint for every candidate:\n'
        f'{json.dumps(factors, ensure_ascii=True, indent=2)}\n'
        'Preserve the original wording and relation direction in source_span. '
        'Do not turn approximate descriptions into stricter requirements or add constraints.\n'
        'For each factor, ask: Does the current image provide clear, significant '
        'evidence that contradicts this constraint?\n'
        'Use exactly one status:\n'
        '- "contradicted": clear visible evidence is incompatible with the constraint. '
        'Name the specific observed conflict, for example a clearly dark-blue cushion '
        'when a white cushion is required.\n'
        '- "not_contradicted": the relevant feature or relation can be inspected and '
        'no significant conflict is visible. This does not establish unique identity.\n'
        '- "unknown": occlusion, blur, viewpoint, image boundaries, or missing reference '
        'objects prevent a reliable check. Explain the missing visual information.\n'
        'Use only the current image. Do not invent hidden features or objects. '
        'Unseen or unconfirmed information is not a contradiction. '
        'An unseen reference object makes its relation unknown; seeing a different '
        'table does not prove that no matching table is nearby. '
        'Do not treat shadows, minor color differences, or uncertain material guesses '
        'as significant contradictions. For quantity constraints, require sufficient '
        'visibility to establish a conflict; a possibly hidden second pillow is unknown. '
        'Judge all factors independently and return every factor even after finding a conflict.\n'
        'Return exactly one JSON object with only "candidates". '
        f'This array must have exactly {count} entries in candidate ID order. '
        'Each entry must contain only integer "candidate_id" and array "constraints". '
        f'Each constraints array must contain exactly these factor IDs in order: {factor_ids}. '
        'Every factor must occur once, with only string fields "factor_id", "status", '
        'and "evidence". Evidence must be one short English sentence stating the specific '
        'visible evidence or what prevents judgment. No overall match decision, scores, '
        'actions, Markdown fences, comments, ellipses, or extra text.\n'
        'FORMAT EXAMPLES (fictional, alternative responses; structure only):\n'
        + '\n\n'.join(examples)
        + '\nThese fictional observations are not evidence about the supplied image. '
        'Follow the structure, not the sample decisions. For an actual contradiction, '
        'name the concrete visible conflict rather than copying a generic example sentence. '
        'Return ONE object for the current image and requested constraints only.'
    )


def verify_candidate_constraints(model_client, rgb, boxes, category, attributes, constraints, *, log_context=None):
    """One batched factor check; unknown is preserved for the caller's veto policy."""
    if not boxes:
        return []
    prompt = build_constraint_prompt(category, attributes, constraints, len(boxes))
    factor_ids = [factor['factor_id'] for factor in constraints]
    schema = _constraint_schema(len(boxes), factor_ids)
    response_format = {'type': 'json_schema', 'json_schema': {
        'name': 'instance_constraints', 'strict': True, 'schema': schema}}
    image = _candidate_image(rgb, boxes)
    log_context = {'verification_id': uuid4().hex, **(log_context or {}),
                   'mode': 'subconstraint', 'category': category, 'candidate_count': len(boxes)}
    for attempt in range(3):
        request = prompt
        if attempt:
            request += (f'\nFormatting correction attempt {attempt}: return one JSON object '
                        f'with exactly {len(boxes)} candidates and factor IDs {factor_ids} '
                        'for each candidate in order. Follow the example structure, '
                        'include all factors, and use only the three allowed status strings.')

        context = {**log_context, 'attempt': attempt + 1}
        try:
            results = _complete_logged_verification(
                model_client, request, image, response_format,
                lambda answer: parse_constraint_results(answer, len(boxes), factor_ids), context)
        except (ValueError, TypeError, KeyError, AttributeError):
            LOG.warning('Invalid subconstraint verification: boxes=%d attempt=%d; retrying format only',
                        len(boxes), attempt + 1)
            continue
        log_instance_verification('result', context=context, candidates=results)
        return results
    log_instance_verification('failed', context=log_context, attempts=3, reason='invalid_response_format')
    raise ModelRequestError('Invalid subconstraint verification JSON after 3 format attempts') from None
