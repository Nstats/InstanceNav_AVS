#!/usr/bin/env python
"""S4 of 实验计划.md: offline test of environment evidence at the STOP decision.

Unit of evaluation: every STOP made after a lock in the six k=1 runs (default and
admission-off configurations). A STOP is correct when the episode succeeded. Each
verification variant re-judges the locked candidate at that moment and either keeps
the STOP or rejects it (a veto). Ground truth decides only the scoring and the oracle
variant E2*; every other input is what the agent had at that time.

Variants (one model call per decision point unless noted):
  V0   the current confirmation prompt (utils.instance_verifier.verify_candidates),
       single candidate, asked twice (V0a, V0b) to measure API nondeterminism.
  V1   the object description only (intrinsic attributes), no surroundings.
  E1   per-fact checklist on the lock frame. Facts are the authored factor table
       (data/instancenav_factors/val10_factors_v1_review.csv): object attributes and
       surroundings relations, each supported / contradicted / not_visible, plus an
       overall is_target answer.
  E2   E1 plus up to three earlier views of the same candidate, selected agent-side:
       earlier boxes whose est_gps lies within 1 m of the lock point, smallest box first.
       Without such views E2 reuses the E1 answer.
  E2o  oracle frame selection for E2: earlier boxes on the same GT instance (upper bound).
  E3   room consistency: target room types inferred once per task from the text only,
       candidate room type inferred from the images only (no description); veto when
       both are known and do not overlap.

Usage (repository root):
  python tools/offline_verifier_eval.py --dry-run
  python tools/offline_verifier_eval.py --limit 12
  python tools/offline_verifier_eval.py \
      --out outputs/analysis_offline_verifier_20261102
API answers are cached in <out>/answers.jsonl, so an interrupted run resumes.
Model call logs (utils.instance_verifier prints) go to <out>/calls.log.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.analyze_verification_set import classify, final_lock, load_set, lock_box  # noqa: E402

FACTOR_FILE = REPO / 'data/instancenav_factors/val10_factors_v1_review.csv'
K1_GROUPS = ('def', 'adm')
HISTORY_RADIUS_M = 1.0
HISTORY_MAX = 3
HISTORY_MIN_GAP = 3            # policy steps between selected history frames
FACT_STATUSES = ('supported', 'contradicted', 'not_visible')
ROOMS = ('bedroom', 'living room', 'dining room', 'kitchen', 'bathroom', 'office', 'hallway',
         'laundry room', 'gym', 'garage', 'outdoor or balcony', 'other', 'unknown')
VARIANTS = ('V0a', 'V0b', 'V1', 'E1', 'E2', 'E2o', 'E3')
VIS_PX = 2000

# Rough value of rejecting one wrong STOP: P(target seen later) by failure class (S3,
# Kaplan-Meier, outputs/analysis_verification_set_20261102.md section 8) times
# P(success | target seen before lock) by configuration. A rejected correct STOP is
# counted as one lost success (conservative).
P_SEEN_LATER = {'ws_samecat_before': 0.43, 'ws_other': 0.64, 'ws_samecat_after': 1.0}
P_CONVERT = {'adm': 0.638, 'def': 0.536}


# --------------------------------------------------------------------------------------
# Decision points (no API)
# --------------------------------------------------------------------------------------
def factor_key(ep):
    """Stable task key of the factor table: <scene>_<goal_object_id>__ep<episode_id>."""
    scene_file, episode_id, goal_object_id = ep['key']
    return f"{scene_file.split('.')[0]}_{goal_object_id}__ep{episode_id}"


def load_factors(path=FACTOR_FILE):
    out = defaultdict(list)
    with open(path, newline='', encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            out[row['episode_key'].strip()].append({
                'id': row['factor_id'].strip(), 'kind': row['kind'].strip(),
                'text': row['source_span'].strip()})
    return dict(out)


def _mark(b):
    """Drawable mark of a box record: ('box', [x0,y0,x1,y1]) or ('center', [x, y])."""
    if b.get('box'):
        return ('box', [int(v) for v in b['box']])
    if b.get('center_px'):
        return ('center', [int(v) for v in b['center_px']])
    return None


def select_history(ep, lock, lbox, frames_with_image, *, oracle=False, k=HISTORY_MAX,
                   radius=HISTORY_RADIUS_M, min_gap=HISTORY_MIN_GAP):
    """Earlier views of the locked candidate, widest (smallest box) first.

    Agent-side: boxes before the lock step whose est_gps lies within `radius` of the lock
    point. Oracle: boxes before the lock step labelled with the lock box's GT instance.
    One box per step; selected steps are at least `min_gap` apart.
    """
    g = lock.get('goal_gps')
    cands = []
    for b in ep['boxes']:
        if b.get('step') is None or b['step'] >= lock['step'] or b['step'] not in frames_with_image:
            continue
        if _mark(b) is None:
            continue
        if oracle:
            if lbox is None or b.get('gt_instance') is None or b.get('gt_instance') != lbox.get('gt_instance'):
                continue
        else:
            if not b.get('est_gps') or not g or math.dist(b['est_gps'], g) > radius:
                continue
        area = b.get('box_area_frac')
        cands.append(((area if area is not None else 1.0), b['step'], b))
    cands.sort(key=lambda x: (x[0], x[1]))
    out = []
    for _, step, b in cands:
        if all(abs(step - s) >= min_gap for s, _ in out):
            out.append((step, b))
        if len(out) == k:
            break
    out.sort(key=lambda x: x[0])
    return [{'step': s, 'mark': _mark(b), 'image': frames_with_image[s],
             'box_area_frac': b.get('box_area_frac')} for s, b in out]


def build_decision_points(data, factors):
    runs = [r for r in sorted(data) if data[r]['group'] in K1_GROUPS]
    # Hard distractor = an instance that caused a same-category wrong stop in >= 2 runs.
    wrong_inst = Counter()
    for r in runs:
        for i, e in data[r]['episodes'].items():
            if classify(e, VIS_PX) in ('ws_samecat_before', 'ws_samecat_after'):
                wrong_inst[(i, final_lock(e)['identity_instance'])] += 1
    points = []
    for r in runs:
        for i, e in sorted(data[r]['episodes'].items()):
            if e['termination'] != 'stop_called' or not e['locks']:
                continue
            L = final_lock(e)
            b = lock_box(e, L)
            frames = {f['step']: f['image'] for f in e['frames'] if f.get('image')}
            fcheck = {f['step']: f['check'] for f in e['frames']}
            if b is None or _mark(b) is None or L['step'] not in frames:
                continue
            same_call = [x for x in e['boxes'] if x['step'] == L['step'] and x.get('confirmation') is not None]
            cls = classify(e, VIS_PX)
            points.append({
                'id': f"{r}/{i:03d}/{L['step']}", 'run': r, 'group': data[r]['group'], 'episode': i,
                'category': e['category'], 'success': bool(e['success']), 'cls': cls,
                'lock_step': L['step'], 'lock_path': L.get('path'), 'mark': _mark(b),
                'image': f"{r}/{frames[L['step']]}", 'frame_check': fcheck.get(L['step']),
                'box_area_frac': b.get('box_area_frac'), 'label': b.get('label'),
                'gt_instance': b.get('gt_instance'),
                'hard_distractor': (not e['success'] and wrong_inst[(i, L['identity_instance'])] >= 2
                                    and cls in ('ws_samecat_before', 'ws_samecat_after')),
                'n_boxes_in_logged_call': len(same_call),
                'logged_answer': b.get('confirmation'),
                'factor_key': factor_key(e),
                'factors': factors.get(factor_key(e), []),
                'attributes': {'intrinsic_attributes': e.get('intrinsic_attributes') or '',
                               'extrinsic_attributes': e.get('extrinsic_attributes') or ''},
                'history': [{**h, 'image': f"{r}/{h['image']}"} for h in select_history(e, L, b, frames)],
                'history_oracle': [{**h, 'image': f"{r}/{h['image']}"}
                                   for h in select_history(e, L, b, frames, oracle=True)],
            })
    return points


# --------------------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------------------
def draw_mark(rgb, mark, label=None):
    image = Image.fromarray(np.asarray(rgb)[..., :3]).convert('RGB')
    d = ImageDraw.Draw(image)
    kind, v = mark
    if kind == 'box':
        x0, y0, x1, y1 = v
        d.rectangle((x0, y0, x1, y1), outline='red', width=3)
        tx, ty = x0 + 3, y0 + 3
    else:
        x, y = v
        d.ellipse((x - 14, y - 14, x + 14, y + 14), outline='red', width=3)
        d.line((x - 6, y, x + 6, y), fill='red', width=2)
        d.line((x, y - 6, x, y + 6), fill='red', width=2)
        tx, ty = x + 16, y - 16
    if label is not None:
        d.text((tx, ty), str(label), fill='red', stroke_width=1, stroke_fill='white')
    return image


def load_rgb(root, rel):
    return np.asarray(Image.open(Path(root) / rel).convert('RGB'))


# --------------------------------------------------------------------------------------
# Prompts and parsers (unit-tested)
# --------------------------------------------------------------------------------------
def _strip_fence(answer):
    return re.sub(r'^```(?:json)?\s*|\s*```$', '', answer.strip(), flags=re.IGNORECASE)


def _loads_object(answer):
    data = json.loads(_strip_fence(answer))
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError('expected a JSON object')
    return data


def build_v1_prompt(category, intrinsic):
    return (
        'Stage: goal confirmation. Independently evaluate this candidate.\n'
        f'Find the specific {category} described below. The red box or red circle marks the candidate object.\n'
        f'Description of the object itself:\n{intrinsic}\n'
        'Decide whether the marked object matches this description of the object itself. '
        'Ignore its surroundings. Matching the object category alone is not sufficient. '
        'If evidence is inadequate, answer unknown.\n'
        'Return only one JSON object {"match": "yes" | "no" | "unknown"}. No other text.'
    )


def parse_v1(answer):
    data = _loads_object(answer)
    if set(data) != {'match'} or data['match'] not in ('yes', 'no', 'unknown'):
        raise ValueError('expected {"match": yes|no|unknown}')
    return data['match']


def build_checklist_prompt(category, attributes, facts, n_images):
    description = '\n'.join(str(attributes.get(k, '')) for k in
                            ('intrinsic_attributes', 'extrinsic_attributes')).strip()
    if n_images == 1:
        intro = 'The red box or red circle in the image marks the candidate object.'
    else:
        intro = (f'The {n_images} images show the same candidate object, marked by a red box or red circle '
                 f'in each. Images 0 to {n_images - 2} are earlier views, usually from farther away; '
                 f'image {n_images - 1} is the current view. Use all images together.')
    listed = [{'id': f['id'], 'type': 'object attribute' if f['kind'] == 'attribute' else 'surroundings',
               'fact': f['text']} for f in facts]
    return (
        f'You check whether a candidate object is the specific target {category} described below.\n'
        f'{intro}\n'
        f'Target description:\n{description}\n'
        f'Facts to check:\n{json.dumps(listed, ensure_ascii=False)}\n'
        'For every fact, choose one status for the marked candidate:\n'
        '- "supported": the images show that this fact holds for the marked candidate. For a surroundings '
        'fact, the reference object is visible next to or around the candidate as described.\n'
        '- "contradicted": the relevant part is clearly visible and the fact does not hold. For a surroundings '
        'fact, the area around the candidate where the reference object should be is visible and the '
        'reference object is not there, or the candidate is clearly in a different kind of place than '
        'described. For an object attribute, the attribute is clearly visible and clearly different.\n'
        '- "not_visible": the relevant part of the candidate or its surroundings is outside the images, '
        'occluded, or too small or unclear to judge.\n'
        'Left/right and front/behind in the description refer to the viewpoint of the photo the description '
        'came from; for such facts judge only whether the reference object is near the candidate. '
        'Judge only the marked candidate and do not attribute other objects\' features to it.\n'
        'Then give "is_target": "yes" if the evidence as a whole indicates that the marked candidate is the '
        'described object, "no" if it indicates a different object, "unknown" if the images are insufficient.\n'
        'Return only one JSON object: {"facts": [{"id": ..., "status": ..., "evidence": "..."}, ...], '
        '"is_target": ...}. List every fact once, in the given order. Each evidence is at most 15 words. '
        'No other text.'
    )


def checklist_schema(fact_ids):
    fact = {'type': 'object', 'properties': {
        'id': {'type': 'string', 'enum': list(fact_ids)},
        'status': {'type': 'string', 'enum': list(FACT_STATUSES)},
        'evidence': {'type': 'string'}},
        'required': ['id', 'status', 'evidence'], 'additionalProperties': False}
    return {'type': 'object', 'properties': {
        'facts': {'type': 'array', 'items': fact, 'minItems': len(fact_ids), 'maxItems': len(fact_ids)},
        'is_target': {'type': 'string', 'enum': ['yes', 'no', 'unknown']}},
        'required': ['facts', 'is_target'], 'additionalProperties': False}


def parse_checklist(answer, fact_ids):
    data = _loads_object(answer)
    if set(data) != {'facts', 'is_target'}:
        raise ValueError('expected facts and is_target')
    if data['is_target'] not in ('yes', 'no', 'unknown'):
        raise ValueError('bad is_target')
    facts = data['facts']
    if not isinstance(facts, list) or [f.get('id') if isinstance(f, dict) else None for f in facts] != list(fact_ids):
        raise ValueError('facts must list every id once, in order')
    for f in facts:
        if set(f) != {'id', 'status', 'evidence'} or f['status'] not in FACT_STATUSES:
            raise ValueError('bad fact entry')
        if not isinstance(f['evidence'], str):
            raise ValueError('evidence must be a string')
    return {'facts': {f['id']: f['status'] for f in facts},
            'evidence': {f['id']: f['evidence'][:200] for f in facts},
            'is_target': data['is_target']}


def build_room_text_prompt(category, attributes):
    description = '\n'.join(str(attributes.get(k, '')) for k in
                            ('intrinsic_attributes', 'extrinsic_attributes')).strip()
    return (
        f'A {category} in a house is described below.\n{description}\n'
        'Based only on this description, which types of room or area could the object be in? '
        f'Choose from {json.dumps(list(ROOMS))}. Give every type that fits the described surroundings, '
        'including both parts of a likely open-plan area. Give ["unknown"] if the description contains no '
        'clue about the room.\nReturn only one JSON object {"rooms": [...]}. No other text.'
    )


def build_room_vision_prompt(n_images):
    where = ('in the image' if n_images == 1 else
             f'in each of the {n_images} images (the same object seen from different places)')
    return (
        f'A red box or red circle {where} marks one object inside a house. Which type of room or area is '
        f'the marked object in? Choose from {json.dumps(list(ROOMS))}. Use the visible furniture and layout '
        'around the marked object. If it stands in an open-plan area that combines two types, give both. '
        'Give ["unknown"] if you cannot tell.\nReturn only one JSON object {"rooms": [...]}. No other text.'
    )


ROOM_SCHEMA = {'type': 'object', 'properties': {
    'rooms': {'type': 'array', 'items': {'type': 'string', 'enum': list(ROOMS)}, 'minItems': 1, 'maxItems': 4}},
    'required': ['rooms'], 'additionalProperties': False}


def parse_rooms(answer):
    data = _loads_object(answer)
    if set(data) != {'rooms'} or not isinstance(data['rooms'], list) or not data['rooms']:
        raise ValueError('expected {"rooms": [...]}')
    rooms = [str(x).strip().lower() for x in data['rooms']]
    if any(x not in ROOMS for x in rooms):
        raise ValueError('unknown room label')
    return sorted(set(rooms))


def _rf(name, schema):
    return {'type': 'json_schema', 'json_schema': {'name': name, 'strict': True, 'schema': schema}}


# --------------------------------------------------------------------------------------
# Decision rules (unit-tested). Each returns True (reject the STOP), False (keep) or None
# (abstain = keep, but counted separately for coverage).
# --------------------------------------------------------------------------------------
def rule_match(answer):
    return None if answer is None else answer != 'yes'


def _fact_counts(result, facts, kind):
    ids = [f['id'] for f in facts if f['kind'] == kind]
    st = [result['facts'][i] for i in ids if i in result['facts']]
    return len(ids), st.count('supported'), st.count('contradicted')


def rule_net(result, facts, kind):
    """Reject when contradicted facts of this kind outnumber supported ones (primary for relations)."""
    if result is None:
        return None
    n, sup, con = _fact_counts(result, facts, kind)
    if n == 0:
        return None
    return con > sup


def rule_any(result, facts, kind):
    if result is None:
        return None
    n, sup, con = _fact_counts(result, facts, kind)
    if n == 0:
        return None
    return con > 0


def rule_overall(result):
    return None if result is None else result['is_target'] == 'no'


def rule_room(target_rooms, candidate_rooms):
    if not target_rooms or not candidate_rooms:
        return None
    t = set(target_rooms) - {'unknown', 'other'}
    c = set(candidate_rooms) - {'unknown', 'other'}
    if not t or not c:
        return None
    return not (t & c)


def rule_or(*decisions):
    if any(d is True for d in decisions):
        return True
    if all(d is None for d in decisions):
        return None
    return False


# --------------------------------------------------------------------------------------
# API layer
# --------------------------------------------------------------------------------------
class DiskCachedClient:
    """ModelClient wrapper with a persistent answer cache keyed by tag, prompt and images."""

    def __init__(self, client, path):
        self.client, self.path = client, Path(path)
        self.lock = threading.Lock()
        self.local = threading.local()
        self.cache = {}
        self.calls = Counter()
        if self.path.exists():
            for line in open(self.path):
                try:
                    rec = json.loads(line)
                    self.cache[rec['key']] = rec['answer']
                except (ValueError, KeyError):
                    continue

    @staticmethod
    def key_of(tag, prompt, images, response_format, max_tokens):
        h = hashlib.sha256()
        h.update(json.dumps([tag, prompt, response_format, max_tokens], sort_keys=True).encode())
        for im in images:
            h.update(str(im.size).encode())
            h.update(hashlib.sha256(im.convert('RGB').tobytes()).digest())
        return h.hexdigest()

    def complete(self, prompt, image=None, *, images=None, response_format=None, validator=None,
                 max_tokens=None, retries=None):
        tag = getattr(self.local, 'tag', '')
        imgs = [image] if image is not None else list(images or [])
        key = self.key_of(tag, prompt, imgs, response_format, max_tokens)
        hit = self.cache.get(key)
        if hit is not None:
            if validator is not None:
                validator(hit)
            self.local.last = hit
            self.calls['cache'] += 1
            return hit
        kwargs = {'response_format': response_format, 'validator': validator, 'max_tokens': max_tokens}
        if retries is not None:
            kwargs['retries'] = retries
        if images is not None:
            answer = self.client.complete(prompt, images=images, **kwargs)
        else:
            answer = self.client.complete(prompt, image, **kwargs)
        with self.lock:
            self.cache[key] = answer
            with open(self.path, 'a') as fh:
                fh.write(json.dumps({'key': key, 'tag': tag, 'answer': answer}, ensure_ascii=False) + '\n')
        self.calls['network'] += 1
        self.local.last = answer
        return answer


def _ask(cc, tag, prompt, images, parse, response_format, max_tokens=1024):
    """Up to three format attempts; the retry wording changes so a bad answer is not replayed."""
    last_error = None
    for attempt in range(3):
        request = prompt if attempt == 0 else (
            prompt + f'\nFormatting correction attempt {attempt}: return exactly the JSON object specified above.')
        parsed = {}

        def validate(answer):
            parsed['v'] = parse(answer)

        cc.local.tag = tag
        try:
            if len(images) == 1:
                answer = cc.complete(request, images[0], response_format=response_format,
                                     validator=validate, max_tokens=max_tokens)
            else:
                answer = cc.complete(request, images=images, response_format=response_format,
                                     validator=validate, max_tokens=max_tokens)
            if 'v' not in parsed:
                validate(answer)
            return parsed['v']
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            last_error = error
            continue
    raise ValueError(f'invalid answer after 3 attempts: {last_error}')


_draw_override = threading.local()


def _install_v0_drawing():
    """Let verify_candidates draw a centre marker for scene-graph locks (thread-local switch)."""
    import utils.instance_verifier as iv
    if getattr(iv, '_s4_patched', False):
        return
    original = iv._candidate_image

    def patched(rgb, boxes):
        center = getattr(_draw_override, 'center', None)
        if center is not None:
            return draw_mark(rgb, ('center', center), label=0)
        return original(rgb, boxes)

    iv._candidate_image = patched
    iv._s4_patched = True


def run_v0(cc, root, p, rep):
    from utils.instance_verifier import parse_matches, verify_candidates
    rgb = load_rgb(root, p['image'])
    kind, v = p['mark']
    box = v if kind == 'box' else [v[0] - 1, v[1] - 1, v[0] + 1, v[1] + 1]
    _draw_override.center = v if kind == 'center' else None
    cc.local.tag = f'V0#{rep}'
    cc.local.last = None
    try:
        verify_candidates(cc, rgb, [box], p['category'], p['attributes'],
                          log_context={'stage': 'goal_confirmation', 's4_point': p['id']})
    finally:
        _draw_override.center = None
    return parse_matches(cc.local.last, 1)[0]


def _point_images(root, p, history):
    imgs = [draw_mark(load_rgb(root, h['image']), h['mark']) for h in history]
    imgs.append(draw_mark(load_rgb(root, p['image']), p['mark']))
    return imgs


def run_checklist(cc, root, p, history, tag):
    facts = p['factors']
    if not facts:
        return None
    imgs = _point_images(root, p, history)
    ids = [f['id'] for f in facts]
    prompt = build_checklist_prompt(p['category'], p['attributes'], facts, len(imgs))
    return _ask(cc, tag, prompt, imgs, lambda a: parse_checklist(a, ids),
                _rf('fact_checklist', checklist_schema(ids)), max_tokens=2048)


def run_v1(cc, root, p):
    img = draw_mark(load_rgb(root, p['image']), p['mark'])
    prompt = build_v1_prompt(p['category'], p['attributes']['intrinsic_attributes'])
    schema = {'type': 'object', 'properties': {'match': {'type': 'string', 'enum': ['yes', 'no', 'unknown']}},
              'required': ['match'], 'additionalProperties': False}
    return _ask(cc, 'V1', prompt, [img], parse_v1, _rf('object_match', schema), max_tokens=256)


def run_room_vision(cc, root, p):
    imgs = _point_images(root, p, p['history'])
    return _ask(cc, 'E3v', build_room_vision_prompt(len(imgs)), imgs, parse_rooms,
                _rf('room_type', ROOM_SCHEMA), max_tokens=256)


def run_room_text(cc, category, attributes):
    prompt = build_room_text_prompt(category, attributes)
    last = None
    for attempt in range(3):
        request = prompt if attempt == 0 else prompt + f'\nFormatting correction attempt {attempt}.'
        parsed = {}
        cc.local.tag = 'E3t'
        try:
            ans = cc.complete(request, None, response_format=_rf('room_type', ROOM_SCHEMA),
                              validator=lambda a: parsed.__setitem__('v', parse_rooms(a)), max_tokens=256)
            return parsed.get('v') or parse_rooms(ans)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            last = error
    raise ValueError(f'invalid answer after 3 attempts: {last}')


def run_all(points, root, out, variants, workers):
    from utils.model_client import ModelClient, ModelRequestError
    client = ModelClient()
    print(f'model={client.model} backend={client.backend} cache_size={client.cache_size}', file=sys.stderr)
    cc = DiskCachedClient(client, out / 'answers.jsonl')
    _install_v0_drawing()
    results = defaultdict(dict)
    errors = []
    room_text = {}
    jobs = []
    if 'E3' in variants:
        for p in points:
            room_text.setdefault(p['factor_key'], (p['category'], p['attributes']))
        jobs += [('E3t', k) for k in room_text]
    for p in points:
        for v in variants:
            if v == 'E3':
                jobs.append(('E3v', p))
            elif v == 'E2' and not p['history']:
                continue                      # reuses E1
            elif v == 'E2o' and not p['history_oracle']:
                continue
            else:
                jobs.append((v, p))

    def do(job):
        v, p = job
        if v == 'E3t':
            cat, attrs = room_text[p]
            return v, p, run_room_text(cc, cat, attrs)
        if v in ('V0a', 'V0b'):
            return v, p, run_v0(cc, root, p, v[-1])
        if v == 'V1':
            return v, p, run_v1(cc, root, p)
        if v == 'E1':
            return v, p, run_checklist(cc, root, p, [], 'E1')
        if v == 'E2':
            return v, p, run_checklist(cc, root, p, p['history'], 'E2')
        if v == 'E2o':
            return v, p, run_checklist(cc, root, p, p['history_oracle'], 'E2o')
        if v == 'E3v':
            return v, p, run_room_vision(cc, root, p)
        raise ValueError(v)

    started = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(do, j): j for j in jobs}
        for fut in as_completed(futs):
            v, p = futs[fut]
            pid = p if isinstance(p, str) else p['id']
            try:
                _, _, res = fut.result()
                if v == 'E3t':
                    results['__room_text__'][pid] = res
                else:
                    results[pid][v] = res
            except (ModelRequestError, ValueError, TypeError, KeyError, AttributeError) as error:
                errors.append({'variant': v, 'point': pid, 'error': f'{type(error).__name__}: {error}'[:300]})
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f'[{done}/{len(jobs)}] {time.time() - started:.0f}s network={cc.calls["network"]} '
                      f'cache={cc.calls["cache"]} errors={len(errors)}', file=sys.stderr, flush=True)
    for p in points:
        r = results[p['id']]
        if 'E2' in variants and not p['history'] and 'E1' in r:
            r['E2'] = r['E1']
        if 'E2o' in variants and not p['history_oracle'] and 'E1' in r:
            r['E2o'] = r['E1']
    return results, errors, dict(cc.calls)


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def decisions_for(p, r, room_text):
    f = p['factors']
    d = {
        'V0a': rule_match(r.get('V0a')), 'V0b': rule_match(r.get('V0b')), 'V1': rule_match(r.get('V1')),
    }
    for v in ('E1', 'E2', 'E2o'):
        res = r.get(v)
        d[f'{v}.rel'] = rule_net(res, f, 'relation')            # primary
        d[f'{v}.rel_any'] = rule_any(res, f, 'relation')
        d[f'{v}.attr'] = rule_net(res, f, 'attribute')
        d[f'{v}.overall'] = rule_overall(res)
    t = room_text.get(p['factor_key'])
    d['E3'] = rule_room(t, r.get('E3v'))
    d['E2.rel|E3'] = rule_or(d['E2.rel'], d['E3'])
    d['E2.overall|E3'] = rule_or(d['E2.overall'], d['E3'])
    d['E2o.rel|E3'] = rule_or(d['E2o.rel'], d['E3'])
    return d


def _cluster_ci(rows, stat, n_boot=2000, seed=0):
    by_ep = defaultdict(list)
    for x in rows:
        by_ep[x['episode']].append(x)
    keys = list(by_ep)
    if len(keys) < 2:
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [x for _ in keys for x in by_ep[keys[rng.randrange(len(keys))]]]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]) if vals else (None, None)


def _rate(rows, key):
    xs = [x for x in rows if x['d'].get(key) is not None or True]
    if not xs:
        return None
    return sum(bool(x['d'].get(key)) for x in xs) / len(xs)


def _gain(rows, key):
    """Estimated SR change per run (100 episodes): recovered wrong stops minus lost correct ones."""
    g = 0.0
    for x in rows:
        if not x['d'].get(key):
            continue
        if x['success']:
            g -= 1.0
        else:
            g += P_SEEN_LATER.get(x['cls'], 0.5) * P_CONVERT[x['group']]
    n_runs = len({x['run'] for x in rows}) or 1
    return g / n_runs


def score(points, results, n_boot=2000):
    room_text = results.get('__room_text__', {})
    rows = []
    for p in points:
        r = results.get(p['id'], {})
        rows.append({**{k: p[k] for k in ('id', 'run', 'group', 'episode', 'category', 'success', 'cls',
                                          'hard_distractor', 'mark', 'n_boxes_in_logged_call')},
                     'center': p['mark'][0] == 'center',
                     'has_history': bool(p['history']), 'has_relations': any(f['kind'] == 'relation' for f in p['factors']),
                     'd': decisions_for(p, r, room_text), 'answered': sorted(r)})
    keys = list(rows[0]['d']) if rows else []
    wrong = [x for x in rows if not x['success']]
    right = [x for x in rows if x['success']]
    table = {}
    for k in keys:
        def wr(s, k=k):
            w = [x for x in s if not x['success']]
            return sum(bool(x['d'][k]) for x in w) / len(w) if w else None

        def cr(s, k=k):
            c = [x for x in s if x['success']]
            return sum(bool(x['d'][k]) for x in c) / len(c) if c else None

        def gain(s, k=k):
            return _gain(s, k)
        covered = sum(x['d'][k] is not None for x in rows)
        table[k] = {
            'wrong_rejected': wr(rows), 'wrong_ci': _cluster_ci(rows, wr, n_boot),
            'correct_rejected': cr(rows), 'correct_ci': _cluster_ci(rows, cr, n_boot),
            'gain_per_run': gain(rows), 'gain_ci': _cluster_ci(rows, gain, n_boot),
            'coverage': covered / len(rows) if rows else None,
            'wrong_rejected_by_cls': {c: (sum(bool(x['d'][k]) for x in wrong if x['cls'] == c),
                                          sum(1 for x in wrong if x['cls'] == c))
                                      for c in sorted({x['cls'] for x in wrong})},
            'hard_distractor_rejected': (sum(bool(x['d'][k]) for x in wrong if x['hard_distractor']),
                                         sum(1 for x in wrong if x['hard_distractor'])),
            'box_points': {'wrong': wr([x for x in rows if not x['center']]),
                           'correct': cr([x for x in rows if not x['center']])},
            'center_points': {'wrong': wr([x for x in rows if x['center']]),
                              'correct': cr([x for x in rows if x['center']])},
        }
    return rows, table, {'n_points': len(rows), 'n_wrong': len(wrong), 'n_correct': len(right)}


def fact_diagnostics(points, results):
    """Per-fact status distribution on correct vs wrong STOPs, by fact kind and variant."""
    out = {}
    for v in ('E1', 'E2', 'E2o'):
        c = defaultdict(Counter)
        for p in points:
            res = results.get(p['id'], {}).get(v)
            if res is None:
                continue
            side = 'correct' if p['success'] else 'wrong'
            for f in p['factors']:
                s = res['facts'].get(f['id'])
                if s:
                    c[(f['kind'], side)][s] += 1
            c[('is_target', side)][res['is_target']] += 1
        out[v] = {f'{k[0]}/{k[1]}': dict(cnt) for k, cnt in c.items()}
    return out


def _auc(pairs):
    """pairs: (score, is_correct). P(score of a wrong stop > score of a correct stop), ties count half."""
    w = [s for s, ok in pairs if not ok]
    c = sorted(s for s, ok in pairs if ok)
    if not w or not c:
        return None
    import bisect
    tot = 0.0
    for x in w:
        lo = bisect.bisect_left(c, x)
        hi = bisect.bisect_right(c, x)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(w) * len(c))


def _fact_score(answer, facts, kind=None, mode='net'):
    if not answer:
        return None
    ids = [f['id'] for f in facts if kind is None or f['kind'] == kind]
    if not ids:
        return None
    st = [answer['facts'][i] for i in ids if i in answer['facts']]
    if mode == 'net':
        return (st.count('contradicted') - st.count('supported')) / len(ids)
    return -st.count('supported') / len(ids)          # 'unsupported': lack of positive evidence


def continuous_scores(p, r):
    """Higher = more likely a wrong stop. Post hoc; for AUC and the best-threshold bound only."""
    f = p['factors']
    yesno = {'yes': 0.0, 'unknown': 0.5, 'no': 1.0}
    out = {}
    if r.get('V0a') is not None:
        out['V0a'] = yesno[r['V0a']]
        if r.get('V0b') is not None:
            out['V0 两次平均'] = (yesno[r['V0a']] + yesno[r['V0b']]) / 2
    if r.get('V1') is not None:
        out['V1'] = yesno[r['V1']]
    for v in ('E1', 'E2', 'E2o'):
        a = r.get(v)
        out[f'{v} 周围物体（矛盾−成立）'] = _fact_score(a, f, 'relation')
        out[f'{v} 周围物体（缺少成立）'] = _fact_score(a, f, 'relation', mode='unsupported')
        out[f'{v} 物体属性'] = _fact_score(a, f, 'attribute')
        out[f'{v} 全部事实'] = _fact_score(a, f)
        out[f'{v} 整体判断'] = None if not a else yesno[a['is_target']]
    return {k: v for k, v in out.items() if v is not None}


def best_threshold_bound(pairs, max_correct=0.05):
    """In-sample optimistic bound: best wrong-reject rate with correct-reject <= max_correct."""
    n_w = sum(not ok for _, ok in pairs)
    n_c = sum(ok for _, ok in pairs)
    best = 0.0
    for t in sorted({s for s, _ in pairs}):
        cr = sum(s >= t for s, ok in pairs if ok) / n_c if n_c else 0
        wr = sum(s >= t for s, ok in pairs if not ok) / n_w if n_w else 0
        if cr <= max_correct:
            best = max(best, wr)
    return best


def exploratory(points, results, root):
    """Post hoc: AUC per signal, room split of wrong stops, and fact visibility."""
    import re as _re
    regions_cache = {}

    def regions(scene_path):
        if scene_path not in regions_cache:
            reg = {}
            for line in open(REPO / scene_path.replace('.basis.glb', '.semantic.txt')).readlines()[1:]:
                m = _re.match(r'(\d+),\w+,"(.*)",(\d+)', line.strip())
                if m:
                    reg[int(m.group(1))] = int(m.group(3))
            regions_cache[scene_path] = reg
        return regions_cache[scene_path]

    side = {}
    for p in points:
        if p['success']:
            side[p['id']] = 'correct'
            continue
        e = json.load(open(Path(root) / p['run'] / f"ep_{p['episode']:03d}.json"))
        reg = regions(e['scene_path'])
        g, t = p.get('gt_instance'), e['target_instance']
        side[p['id']] = ('wrong_unknown' if g is None or reg.get(g) is None else
                         'wrong_other_room' if reg.get(g) != reg.get(t) else 'wrong_same_room')
    scores = defaultdict(list)
    for p in points:
        for k, s in continuous_scores(p, results.get(p['id'], {})).items():
            scores[k].append((s, p['success']))
    auc = {k: {'auc': _auc(v), 'n': len(v), 'best_wrong_reject_at_correct_le_5pct': best_threshold_bound(v)}
           for k, v in scores.items()}
    room_text = results.get('__room_text__', {})
    by_side = defaultdict(list)
    for p in points:
        by_side[side[p['id']]].append(p)
    split = {}
    for key in ('V0a', 'V1', 'E1.rel', 'E2.rel', 'E2o.rel', 'E2.attr', 'E2.overall', 'E3', 'E2.rel|E3'):
        split[key] = {}
        for s, ps in by_side.items():
            ds = [decisions_for(p, results.get(p['id'], {}), room_text).get(key) for p in ps]
            split[key][s] = (sum(bool(d) for d in ds), len(ds))
    vis = {}
    for v in ('E1', 'E2', 'E2o'):
        cnt = defaultdict(Counter)
        for p in points:
            a = results.get(p['id'], {}).get(v)
            if not a:
                continue
            for f in p['factors']:
                if f['kind'] == 'relation':
                    cnt[side[p['id']]][a['facts'][f['id']]] += 1
        vis[v] = {s: dict(c) for s, c in cnt.items()}
    return {'auc': auc, 'room_split': split, 'relation_status_by_side': vis,
            'n_by_side': {s: len(ps) for s, ps in by_side.items()}}


def v0_repeatability(points, results):
    agree = Counter()
    for p in points:
        r = results.get(p['id'], {})
        a, b = r.get('V0a'), r.get('V0b')
        if a is not None and b is not None:
            agree['same' if a == b else 'different'] += 1
        if a is not None and p['n_boxes_in_logged_call'] == 1 and p['mark'][0] == 'box':
            agree[f'single_box_vs_logged_{"same" if a == p["logged_answer"] else "different"}'] += 1
    return dict(agree)


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
LABELS = {
    'V0a': 'V0 现有提示词（第 1 次）', 'V0b': 'V0 现有提示词（第 2 次）', 'V1': 'V1 只看物体本身',
    'E1.rel': 'E1 周围物体核对（主规则）', 'E1.rel_any': 'E1 任一周围物体矛盾', 'E1.attr': 'E1 物体属性核对',
    'E1.overall': 'E1 逐项核对后整体判断',
    'E2.rel': 'E2 加历史远景·周围物体（主规则）', 'E2.rel_any': 'E2 任一周围物体矛盾', 'E2.attr': 'E2 物体属性核对',
    'E2.overall': 'E2 逐项核对后整体判断',
    'E2o.rel': 'E2* 真值选远景·周围物体', 'E2o.rel_any': 'E2* 任一周围物体矛盾', 'E2o.attr': 'E2* 物体属性核对',
    'E2o.overall': 'E2* 逐项核对后整体判断',
    'E3': 'E3 房间一致性', 'E2.rel|E3': 'E2 主规则 或 E3', 'E2.overall|E3': 'E2 整体判断 或 E3',
    'E2o.rel|E3': 'E2* 主规则 或 E3',
}


def _p(x):
    return '—' if x is None else f'{100 * x:.1f}%'


def _ci(ci):
    return '' if not ci or ci[0] is None else f' [{100 * ci[0]:.0f}, {100 * ci[1]:.0f}]'


def write_tables(table, meta, rep, diag, calls, errors, path):
    L = ['# S4 自动生成表格', '',
         f"决定点 {meta['n_points']} 个：停错 {meta['n_wrong']}、停对 {meta['n_correct']}。区间为按任务的 95% bootstrap 区间。"
         f"每次运行的 SR 增益按“拒掉错停 × 之后找到并成功的概率 − 拒掉正确停止 × 1”粗估。", '',
         '| 方式 | 拒掉的错停 | 误拒的正确停止 | 每次运行的 SR 增益（点） | 覆盖率 | 难干扰物错停被拒 |',
         '|---|---|---|---|---|---|']
    for k, v in table.items():
        hd = v['hard_distractor_rejected']
        g = v['gain_per_run']
        gci = v['gain_ci']
        L.append(f"| {LABELS.get(k, k)} | {_p(v['wrong_rejected'])}{_ci(v['wrong_ci'])} | "
                 f"{_p(v['correct_rejected'])}{_ci(v['correct_ci'])} | "
                 f"{g:+.1f}" + (f" [{gci[0]:+.1f}, {gci[1]:+.1f}]" if gci and gci[0] is not None else '') +
                 f" | {_p(v['coverage'])} | {hd[0]}/{hd[1]} |")
    L += ['', '## 按错停类型', '', '| 方式 | ' + ' | '.join(sorted(next(iter(table.values()))['wrong_rejected_by_cls'])) + ' |',
          '|---|' + '---|' * len(next(iter(table.values()))['wrong_rejected_by_cls'])]
    for k, v in table.items():
        L.append(f"| {LABELS.get(k, k)} | " + ' | '.join(f'{a}/{b}' for a, b in v['wrong_rejected_by_cls'].values()) + ' |')
    L += ['', '## 有框 / 只有中心点的决定点', '', '| 方式 | 有框：拒错停 | 有框：误拒 | 中心点：拒错停 | 中心点：误拒 |', '|---|---|---|---|---|']
    for k, v in table.items():
        L.append(f"| {LABELS.get(k, k)} | {_p(v['box_points']['wrong'])} | {_p(v['box_points']['correct'])} | "
                 f"{_p(v['center_points']['wrong'])} | {_p(v['center_points']['correct'])} |")
    L += ['', '## V0 可重复性', '', json.dumps(rep, ensure_ascii=False), '', '## 逐项核对的状态分布', '']
    for v, d in diag.items():
        L.append(f'### {v}')
        for k, cnt in sorted(d.items()):
            tot = sum(cnt.values())
            L.append(f"- {k}: " + '，'.join(f'{s} {n}（{100 * n / tot:.0f}%）' for s, n in sorted(cnt.items())))
        L.append('')
    L += ['## 调用', '', json.dumps(calls), f'失败 {len(errors)} 次']
    Path(path).write_text('\n'.join(L) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/verification_set')
    ap.add_argument('--results-root', default='data')
    ap.add_argument('--factors', default=str(FACTOR_FILE))
    ap.add_argument('--out', default='outputs/analysis_offline_verifier')
    ap.add_argument('--variants', nargs='*', default=list(VARIANTS))
    ap.add_argument('--limit', type=int, help='Only the first N decision points (stratified smoke test)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--dry-run', action='store_true', help='Build decision points and count calls; no API')
    args = ap.parse_args(argv)
    root = REPO / args.root
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    data = load_set(root, REPO / args.results_root)
    points = build_decision_points(data, load_factors(args.factors))
    if args.limit:
        rng = random.Random(0)
        wrong = [p for p in points if not p['success']]
        right = [p for p in points if p['success']]
        rng.shuffle(wrong)
        rng.shuffle(right)
        points = wrong[:(args.limit + 1) // 2] + right[:args.limit // 2]
    with open(out / 'decision_points.jsonl', 'w') as fh:
        for p in points:
            fh.write(json.dumps({k: v for k, v in p.items() if k != 'attributes'}, ensure_ascii=False) + '\n')
    n_hist = sum(bool(p['history']) for p in points)
    n_or = sum(bool(p['history_oracle']) for p in points)
    n_rel = sum(any(f['kind'] == 'relation' for f in p['factors']) for p in points)
    summary = {'points': len(points), 'wrong': sum(not p['success'] for p in points),
               'correct': sum(p['success'] for p in points),
               'center_points': sum(p['mark'][0] == 'center' for p in points),
               'with_history': n_hist, 'with_oracle_history': n_or, 'with_relations': n_rel,
               'tasks': len({p['factor_key'] for p in points})}
    est = {'V0a': len(points), 'V0b': len(points), 'V1': len(points), 'E1': len(points), 'E2': n_hist,
           'E2o': n_or, 'E3': len(points) + summary['tasks']}
    summary['estimated_calls'] = sum(est[v] for v in args.variants)
    print(json.dumps(summary, ensure_ascii=False), file=sys.stderr)
    if args.dry_run:
        return summary
    os.environ.setdefault('QWEN_MODEL_NAME', 'qwen3.5-flash')
    os.environ.setdefault('PSGNAV_MODEL_BACKEND', 'openai')
    os.environ.setdefault('PSGNAV_ENV_FILE', str(REPO / '.env'))
    os.environ['PSGNAV_API_CONCURRENCY'] = str(args.workers)
    os.environ['PSGNAV_API_CACHE_SIZE'] = '0'       # V0b must be a fresh request
    os.environ.setdefault('PSGNAV_API_MAX_TOKENS', '4096')
    log = open(out / 'calls.log', 'a')
    real_stdout = sys.stdout
    sys.stdout = log                                 # verify_candidates prints its log records
    try:
        results, errors, calls = run_all(points, root, out, args.variants, args.workers)
    finally:
        sys.stdout = real_stdout
        log.close()
    rows, table, meta = score(points, results, args.boot)
    rep = v0_repeatability(points, results)
    diag = fact_diagnostics(points, results)
    expl = exploratory(points, results, root)
    meta['unique_inputs'] = len({(p['episode'], hashlib.md5(open(root / p['image'], 'rb').read()).hexdigest(),
                                  json.dumps(p['mark'])) for p in points})
    (out / 'results.json').write_text(json.dumps(
        {'summary': summary, 'meta': meta, 'table': table, 'v0_repeatability': rep, 'fact_diagnostics': diag,
         'exploratory': expl, 'calls': calls, 'errors': errors, 'room_text': results.get('__room_text__', {})},
        ensure_ascii=False, indent=1))
    L = ['', '## 事后分析（规则未预先固定，只作参考）', '',
         f"不同的输入（同一任务、同一画面、同一标记）共 {meta['unique_inputs']} 个；同一任务在不同运行中走出完全相同的轨迹时，决定点会重复。", '',
         '| 信号 | AUC | 误拒正确停止 ≤5% 时最多拒掉的错停（样本内最优阈值） |', '|---|---|---|']
    for k, v in expl['auc'].items():
        L.append(f"| {k} | {v['auc']:.3f} | {_p(v['best_wrong_reject_at_correct_le_5pct'])} |")
    sides = ['wrong_other_room', 'wrong_same_room', 'correct']
    L += ['', '按难干扰物是否与目标同房间（HM3D 区域）：', '', '| 方式 | ' + ' | '.join(sides) + ' |', '|---|---|---|---|']
    for k, v in expl['room_split'].items():
        L.append(f'| {LABELS.get(k, k)} | ' + ' | '.join(f'{v[s][0]}/{v[s][1]}' for s in sides if s in v) + ' |')
    L += ['', '周围物体事实的状态分布：', '']
    for v, d in expl['relation_status_by_side'].items():
        for s in sides:
            c = d.get(s, {})
            tot = sum(c.values()) or 1
            L.append(f"- {v} {s}: " + '，'.join(f'{k} {100 * n / tot:.0f}%' for k, n in sorted(c.items())) + f'（{tot}）')
    with open(out / 'per_point.jsonl', 'w') as fh:
        for x in rows:
            r = results.get(x['id'], {})
            fh.write(json.dumps({**x, 'answers': r}, ensure_ascii=False) + '\n')
    write_tables(table, meta, rep, diag, calls, errors, out / 'tables.md')
    with open(out / 'tables.md', 'a') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'wrote {out}', file=sys.stderr)
    return table


if __name__ == '__main__':
    main()
