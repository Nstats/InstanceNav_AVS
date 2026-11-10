"""The frozen S7 pairwise prompt and its online tournament (no evaluation imports)."""
from io import BytesIO
import json
import re

from PIL import Image, ImageDraw

from utils.log_utils import log_event
from utils.model_client import ModelRequestError

CHOICES = ('A', 'B', 'same_object', 'unsure')
SCHEMA = {'type': 'object', 'properties': {'choice': {'type': 'string', 'enum': list(CHOICES)}},
          'required': ['choice'], 'additionalProperties': False}


def build_pair_prompt(category, intrinsic, extrinsic):
    return (
        'Stage: candidate comparison.\n'
        f'Two images are given. In image A and in image B, a red box (or red circle) marks one candidate {category}. '
        f'The task is to find one specific {category} in this house, described below. '
        'The two marked candidates may be different objects or, occasionally, the same object seen twice.\n'
        f'Description of the object itself:\n{intrinsic}\n'
        f'Description of its surroundings:\n{extrinsic}\n'
        'Compare the two marked candidates with each other against the description, using both the '
        "object's own appearance and its visible surroundings. Decide which marked candidate is more likely "
        f'to be the described {category}. Answer "same_object" only if both images clearly show the same '
        'physical object, and "unsure" only if the images give no basis to prefer either.\n'
        'Return only one JSON object {"choice": "A" | "B" | "same_object" | "unsure"}. No other text.'
    )


def parse_pair(answer):
    from utils.instance_verifier import _unique_json_object
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', answer.strip(), flags=re.IGNORECASE)
    data = json.loads(text, object_pairs_hook=_unique_json_object)
    if not isinstance(data, dict) or set(data) != {'choice'} or data['choice'] not in CHOICES:
        raise ValueError('expected {"choice": A|B|same_object|unsure}')
    return data['choice']


def marked_view(candidate, label):
    view = candidate.best_view
    if view is None:
        raise ValueError(f'Candidate {candidate.candidate_id} has no yes view')
    with Image.open(BytesIO(view['png'])) as source:
        image = source.convert('RGB')
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = view['box']
    draw.rectangle((x0, y0, x1, y1), outline='red', width=3)
    draw.text((x0 + 3, y0 + 3), label, fill='red', stroke_width=1, stroke_fill='white')
    return image


def compare_candidates(client, candidates, category, attributes, *, context=None):
    """Input sorted by descending q, then stable ID. Never turn request errors into ties."""
    if not 1 <= len(candidates) <= 3:
        raise ValueError('Comparison requires one to three candidates')
    if len(candidates) == 1:
        return candidates[0], {'logical_requests': 0, 'format_retries': 0, 'scores': [0.]}
    prompt = build_pair_prompt(category, attributes.get('intrinsic_attributes', ''),
                               attributes.get('extrinsic_attributes', ''))
    response_format = {'type': 'json_schema', 'json_schema': {
        'name': 'pair_choice', 'strict': True, 'schema': SCHEMA}}
    points = [0.] * len(candidates)
    stats = {'logical_requests': 0, 'format_retries': 0}
    context = context or {}
    for a in range(len(candidates)):
        for b in range(a + 1, len(candidates)):
            answers = []
            for first, second in ((a, b), (b, a)):
                stats['logical_requests'] += 1
                images = [marked_view(candidates[first], 'A'), marked_view(candidates[second], 'B')]
                for attempt in range(3):
                    request = prompt if not attempt else prompt + (
                        f'\nFormatting correction attempt {attempt}: return only one valid choice JSON object.')
                    try:
                        answer = client.complete(request, images=images, response_format=response_format,
                                                 max_tokens=64)
                        log_event('Relative Pair Response', **context, first=candidates[first].candidate_id,
                                  second=candidates[second].candidate_id, attempt=attempt + 1, answer=answer)
                        choice = parse_pair(answer)
                    except (ValueError, TypeError, KeyError, AttributeError) as exc:
                        log_event('Relative Pair Format Error', **context, attempt=attempt + 1, error=str(exc))
                        if attempt == 2:
                            raise ModelRequestError('Invalid relative-evidence comparison after 3 format attempts') from exc
                        stats['format_retries'] += 1
                    else:
                        answers.append(choice)
                        break
            if answers == ['A', 'B']:
                points[a] += 1
            elif answers == ['B', 'A']:
                points[b] += 1
            else:
                points[a] += .5
                points[b] += .5
    winner = max(range(len(candidates)), key=lambda i: (points[i], -i))
    stats['scores'] = points
    return candidates[winner], stats
