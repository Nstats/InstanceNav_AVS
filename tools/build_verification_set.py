#!/usr/bin/env python
"""Build an offline instance-verification set from InstanceNav run logs.

For every episode of the selected runs, this re-renders the frames the agent saw at the
logged poses, then labels each logged detection box with its ground-truth scene instance
using the semantic sensor. It also records which same-category instances were visible at
every policy step ("exposure"), and the per-box admission/confirmation answers recovered
from the verification log events.

Ground truth here is evaluation-only. Nothing produced by this tool is read by the policy.

Collect-only runs (INSTANCE_COLLECT_ONLY, 实验计划.md S6) never lock. Their logs add:
  * [Collect Frame]: every kept box of a verified frame by box index, with its candidate
    track id and position estimate. It is written just before that frame's filter_result
    and is the authoritative source of kept-box geometry (far kept boxes have no
    [Detection] line). Boxes gain track_id / collect_* fields.
  * [Intended Lock]: the lock the baseline would have made. It is stored under
    intended_locks (never mixed into locks) and labelled through the box of the same
    track at the same step; collect_tracks gives each track's majority GT instance.
  * [Collect Snapshot] and [Landmark Utility]: agent-side state, copied through unchanged.

Frame alignment (see habitat-lab/habitat/core/benchmark.py):
  * act(t) receives the observation produced after the primitives of step t-1.
  * [Step] t logs agent_world/heading_deg from that observation (full_pose is set at the
    start of act), and goal_visible_px from the observation produced *after* step t.
  * So the frame seen at policy step t is rendered at the pose of [Step] t, with the camera
    tilt accumulated from the actions of steps 1..t-1, and its target pixel count must equal
    goal_visible_px of [Step] t-1. That equality is the per-frame consistency check.

Pose conventions (habitat EpisodicGPSSensor / EpisodicCompassSensor):
  gps = [-z, x] of the agent position in the episode start frame,
  compass = theta with agent_rotation = start_rotation * R_y(theta),
  full_pose = [20 + gps[0], 20 - gps[1], degrees(compass)]  (utils/map_manager.py).

Usage (from the repository root, inside the psgnav-instance environment):
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=habitat-lab:. $PSGNAV_PYTHON tools/build_verification_set.py \
      --runs 20261030_211522_2605603 20261031_133130_3729343 --gpu 0
Outputs go to data/verification_set/<run>/ (one JSON per episode, JPEG frames for
detection steps, merged boxes.jsonl), plus data/verification_set/manifest.json.
"""
import argparse
import gzip
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MAP_HALF_M = 20.0          # map_size_cm / 100 / 2 in utils/map_manager.py
TILT_STEP_DEG = 30         # SIMULATOR.TILT_ANGLE
LOOK_UP, LOOK_DOWN = 4, 5  # HabitatSimActions indices (habitat 0.2.1)
NEAR_DISTANCE_M = 4.0      # PSG_Nav_Agent.distance_threshold: near-range confirmation cut-off
VERIFY_STAGES = ('candidate_admission', 'goal_confirmation')
GOAL_CATEGORIES = ('chair', 'sofa', 'bed', 'toilet', 'plant', 'tv_monitor')
STRUCTURAL_NAMES = {'wall', 'floor', 'ceiling', 'unknown', 'misc', 'void', 'objects', ''}


# --------------------------------------------------------------------------------------
# Pose math (numpy-quaternion only; unit-tested in tests/test_verification_set.py)
# --------------------------------------------------------------------------------------
def coeff_to_quat(coeffs):
    import quaternion  # noqa: F401  (registers np.quaternion)
    x, y, z, w = coeffs
    return np.quaternion(w, x, y, z)


def rotate_vector(q, v):
    import quaternion  # noqa: F401
    vq = np.quaternion(0.0, *[float(c) for c in v])
    return (q * vq * q.inverse()).imag


def yaw_quat(theta):
    import quaternion  # noqa: F401
    return np.quaternion(np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0)


def map_pose_to_world(agent_world, heading_deg, start_position, start_rotation, half=MAP_HALF_M):
    """Logged full_pose (map frame, metres/degrees) -> world position (y from start) and rotation."""
    gps0 = float(agent_world[0]) - half
    gps1 = half - float(agent_world[1])
    local = np.array([gps1, 0.0, -gps0])
    q0 = coeff_to_quat(start_rotation)
    position = np.asarray(start_position, dtype=float) + rotate_vector(q0, local)
    rotation = q0 * yaw_quat(np.deg2rad(float(heading_deg)))
    return position, rotation


def world_to_map_pose(position, rotation, start_position, start_rotation, half=MAP_HALF_M):
    """Inverse of map_pose_to_world, using the habitat GPS/compass sensor formulas."""
    q0 = coeff_to_quat(start_rotation)
    v = rotate_vector(q0.inverse(), np.asarray(position, dtype=float) - np.asarray(start_position, dtype=float))
    gps = (-v[2], v[0])
    hv = rotate_vector(rotation.inverse() * q0, (0.0, 0.0, -1.0))
    phi = np.arctan2(hv[0], -hv[2])
    return [half + gps[0], half - gps[1]], float(np.rad2deg(phi))


def world_to_gps(point, start_position, start_rotation):
    q0 = coeff_to_quat(start_rotation)
    v = rotate_vector(q0.inverse(), np.asarray(point, dtype=float) - np.asarray(start_position, dtype=float))
    return np.array([-v[2], v[0]])


def angle_diff_deg(a, b):
    return float((a - b + 180.0) % 360.0 - 180.0)


def tilt_delta(step_record):
    """Camera tilt change (degrees) caused by the primitives executed at this policy step."""
    n = int(step_record.get('primitives', 1) or 0)
    action = step_record.get('action')
    if action == LOOK_DOWN:
        return -TILT_STEP_DEG * n
    if action == LOOK_UP:
        return TILT_STEP_DEG * n
    return 0


def frame_tilts(steps):
    """{policy_step: tilt of the frame seen at that step} from the logged action sequence."""
    tilt, out = 0, {}
    for t in sorted(steps):
        out[t] = tilt
        tilt += tilt_delta(steps[t])
    return out


# --------------------------------------------------------------------------------------
# Log parsing
# --------------------------------------------------------------------------------------
def _json_body(line):
    try:
        return json.loads(line.partition('] ')[2])
    except json.JSONDecodeError:
        return None


def _new_episode(kv, gpu):
    return {
        'dataset_index': int(kv['dataset_index']), 'gpu': gpu,
        'key': [kv['scene_id'].split('/')[-1], kv['episode_id'], kv['goal_object_id']],
        'scene_path': kv['scene_id'], 'pixel_id': None, 'goal_world_pos': None,
        'steps': {}, 'groups': [], 'loose_dets': [], 'locks': [], 'unlocks': [], 'end': None,
        'intended_locks': [], 'collect_snapshots': [], 'landmark_utility': [], 'collect_orphans': 0,
    }


def parse_log_lines(lines, gpu=None):
    """Parse one worker log into ({dataset_index: episode}, header toggles).

    Verification structure per frame: stage `result` events, then one `filter_result`
    (stage candidate_pipeline) that starts a group, then the [Detection] lines of that
    frame. Rejected boxes are emitted in box-index order; kept boxes afterwards.
    In collect-only runs a [Collect Frame] line precedes the filter_result of its frame.
    """
    episodes, toggles = {}, {}
    cur, group, pending = None, None, []
    collect = None   # [Collect Frame] waiting for the filter_result of the same step

    def drop_collect():
        nonlocal collect
        if collect is not None and cur is not None:
            cur['collect_orphans'] += 1
        collect = None

    for line in lines:
        if line.startswith('[Run Header] '):
            body = _json_body(line) or {}
            toggles.update(body.get('toggles', {}))
            continue
        if line.startswith('[Schedule] event=claim'):
            drop_collect()
            kv = dict(re.findall(r'(\w+)=(\S+)', line))
            cur = _new_episode(kv, gpu)
            episodes[cur['dataset_index']] = cur   # a later claim (retry) replaces an earlier one
            group, pending = None, []
            continue
        if cur is None or not line.startswith('['):
            continue
        tag = line[1:line.find(']')]
        if tag == 'Step':
            d = _json_body(line)
            if d and d.get('policy_step') is not None:
                cur['steps'][int(d['policy_step'])] = {
                    k: d.get(k) for k in ('action', 'primitives', 'agent_world', 'heading_deg',
                                          'goal_visible_px', 'found_goal', 'found_possible_goal',
                                          'in_stuck_recovery')}
            group, pending = None, []
            drop_collect()
        elif tag == 'Instance Verification':
            d = _json_body(line)
            if not d:
                continue
            ev = d.get('event')
            if ev == 'result' and d.get('stage') in VERIFY_STAGES:
                pending.append({'stage': d['stage'], 'step': d.get('step'),
                                'box_indices': d.get('box_indices') or [],
                                'matches': d.get('matches') or [],
                                'verification_id': d.get('verification_id')})
            elif ev == 'filter_result':
                group = {'step': d.get('step'), 'verification_id': d.get('verification_id'),
                         'filter': {k: d.get(k) for k in (
                             'checked_box_indices', 'contradicted_box_indices', 'admitted_box_indices',
                             'confirmation_box_indices', 'confirmed_box_indices', 'kept_box_indices',
                             'rejected_box_indices')},
                         'stage_results': [p for p in pending if p['step'] == d.get('step')],
                         'dets': [], 'collect': None}
                if collect is not None and collect.get('step') == d.get('step'):
                    group['collect'] = collect.get('boxes') or []
                    collect = None
                drop_collect()
                cur['groups'].append(group)
                pending = []
        elif tag == 'Detection':
            d = _json_body(line)
            if not d:
                continue
            if group is not None and group['step'] == d.get('step'):
                group['dets'].append(d)
            else:
                cur['loose_dets'].append(d)
        elif tag == 'Goal Mapping':
            d = _json_body(line) or {}
            cur['pixel_id'] = d.get('pixel_id')
            cur['goal_world_pos'] = d.get('goal_world_pos')
        elif tag == 'Goal Lock':
            d = _json_body(line)
            if d:
                cur['locks'].append(d)
        elif tag == 'Goal Unlock':
            d = _json_body(line)
            if d:
                cur['unlocks'].append(d)
        elif tag == 'Collect Frame':
            d = _json_body(line)
            if d:
                drop_collect()
                collect = d
        elif tag in ('Intended Lock', 'Collect Snapshot', 'Landmark Utility'):
            d = _json_body(line)
            if d:
                key = {'Intended Lock': 'intended_locks', 'Collect Snapshot': 'collect_snapshots',
                       'Landmark Utility': 'landmark_utility'}[tag]
                d.pop('episode', None)   # agent-side episode counter, redundant here
                cur[key].append(d)
        elif tag == 'EpisodeEnd':
            cur['end'] = _json_body(line)
            group, pending = None, []
            drop_collect()
    return episodes, toggles


def parse_run(tag, logs_dir):
    paths = sorted(Path(logs_dir).glob(f'instancenav_val10_100_{tag}_gpu*.log'))
    if not paths:
        raise FileNotFoundError(f'No worker logs for run {tag} in {logs_dir}')
    episodes, toggles = {}, {}
    for path in paths:
        gpu = int(re.search(r'_gpu(\d+)\.log$', path.name).group(1))
        with open(path, errors='replace') as fh:
            eps, tg = parse_log_lines(fh, gpu=gpu)
        toggles.update(tg)
        for idx, ep in eps.items():
            if ep['end'] is not None or idx not in episodes:
                episodes[idx] = ep
    return {i: e for i, e in episodes.items() if e['end'] is not None}, toggles


def _assign_kept_lines(kept, kept_dets, filt, anchors):
    """Place re-emitted kept [Detection] lines when some kept boxes were skipped.

    Constraints: a kept box was sent to near-range confirmation iff it is in
    confirmation_box_indices, and its line then has est_gps and est_distance_m below
    NEAR_DISTANCE_M (same geometry call); lines are emitted in increasing box index;
    prediction indices increase with box index, so rejected boxes act as rank anchors.
    A line is assigned only when these constraints leave exactly one index.
    """
    confirm = set(filt.get('confirmation_box_indices') or [])

    def line_is_near(d):
        dist = d.get('est_distance_m')
        return d.get('est_gps') is not None and dist is not None and dist < NEAR_DISTANCE_M

    anchored = [(ri, rd.get('prediction_idx')) for ri, rd in anchors if rd.get('prediction_idx') is not None]

    def rank_ok(i, p):
        if p is None:
            return True
        return sum(ri < i for ri, _ in anchored) == sum(rp < p for _, rp in anchored)

    out, last = {}, -1
    for pos, d in enumerate(kept_dets):
        near = line_is_near(d)
        cands = [i for i in kept if i > last and i not in out and ((i in confirm) == near)
                 and rank_ok(i, d.get('prediction_idx'))]
        # Later lines need later indices: drop candidates that leave too few for them.
        later = kept_dets[pos + 1:]
        cands = [i for i in cands if sum(j > i for j in kept) >= len(later)]
        if len(cands) == 1:
            out[cands[0]] = d
            last = cands[0]
    return out


def _det_matches_box(det, box):
    """A kept [Detection] line belongs to a [Collect Frame] box: same box, or (scene-graph
    lines that log only the mask centre) centre inside the box."""
    if box is None:
        return False
    if det.get('box') is not None:
        return [int(v) for v in det['box']] == [int(v) for v in box]
    c = det.get('center_px')
    return c is not None and box[0] <= c[0] < box[2] and box[1] <= c[1] < box[3]


def group_boxes(group):
    """Per-box records of one verification group, with stage answers and box coordinates.

    Returns (boxes, alignment_ok). alignment_ok is False when some box cannot be given
    its logged line: without a [Collect Frame], when the number of [Detection] lines does
    not match the filter result (coordinates are then attached only where the order is
    unambiguous); with one, only when rejected lines are missing or a kept box is absent
    from it.
    """
    f = group['filter']
    rejected = sorted(f.get('rejected_box_indices') or [])
    kept = list(f.get('kept_box_indices') or [])
    n = len(rejected) + len(kept)
    answers = defaultdict(dict)
    for res in group['stage_results']:
        for j, i in enumerate(res['box_indices']):
            if j < len(res['matches']):
                answers[i][res['stage']] = res['matches'][j]
    rej_dets = [d for d in group['dets'] if not d.get('kept')]
    kept_dets = [d for d in group['dets'] if d.get('kept')]
    collect = {c['box_index']: c for c in (group.get('collect') or []) if c.get('box_index') is not None}
    det_of = {}
    rej_ok = len(rej_dets) == len(rejected)
    if rej_ok:
        det_of.update(zip(rejected, rej_dets))
    if group.get('collect') is not None:
        # Collect-only run: kept boxes are listed by index, so their geometry is exact.
        # [Detection] lines only add source/prediction_idx/centre where they match a box.
        kept_ok = all(i in collect for i in kept)
        used = set()
        for i in kept:
            c = collect.get(i)
            k = next((k for k, d in enumerate(kept_dets)
                      if k not in used and c is not None and _det_matches_box(d, c.get('box'))), None)
            if k is not None:
                used.add(k)
                det_of[i] = kept_dets[k]
    elif len(kept_dets) == len(kept):
        kept_ok = True
        det_of.update(zip(kept, kept_dets))
    else:
        # Kept boxes whose candidate track is unavailable are not re-emitted, so the
        # remaining lines must be placed individually (see _assign_kept_lines).
        kept_ok = False
        if rej_ok:
            det_of.update(_assign_kept_lines(kept, kept_dets, f, list(zip(rejected, rej_dets))))
    boxes = []
    for i in range(n):
        det = det_of.get(i) or {}
        c = collect.get(i) if i in kept else None
        boxes.append({
            'box_index': i, 'kept': i in kept,
            'admission': answers[i].get('candidate_admission'),
            'confirmation': answers[i].get('goal_confirmation'),
            'confirmation_called': i in (f.get('confirmation_box_indices') or []),
            'contradicted': i in (f.get('contradicted_box_indices') or []),
            'box': c.get('box') if c else det.get('box'),
            'center_px': det.get('center_px'),
            'source': det.get('source'),
            'prediction_idx': det.get('prediction_idx'),
            'est_gps': c.get('gps') if c else det.get('est_gps'),
            'est_distance_m': c.get('distance_m') if c else det.get('est_distance_m'),
            'track_id': c.get('track_id') if c else None,
            'collect_confirmed': c.get('confirmed') if c else None,
            'collect_positive': c.get('positive') if c else None,
            'collect_retired': c.get('retired') if c else None,
        })
    return boxes, rej_ok and kept_ok


def episode_box_records(ep):
    """All boxes of an episode: verified groups plus unverified (loose) detections."""
    records, misaligned = [], 0
    for g in ep['groups']:
        boxes, ok = group_boxes(g)
        misaligned += 0 if ok else 1
        for b in boxes:
            b.update(step=g['step'], group_verification_id=g['verification_id'], verified=True)
            records.append(b)
    for d in ep['loose_dets']:
        records.append({'step': d.get('step'), 'box_index': None, 'kept': bool(d.get('kept')),
                        'admission': None, 'confirmation': None, 'confirmation_called': False,
                        'contradicted': False, 'box': d.get('box'), 'center_px': d.get('center_px'),
                        'source': d.get('source'),
                        'prediction_idx': d.get('prediction_idx'), 'est_gps': d.get('est_gps'),
                        'est_distance_m': d.get('est_distance_m'), 'track_id': None,
                        'collect_confirmed': None, 'collect_positive': None, 'collect_retired': None,
                        'group_verification_id': None, 'verified': False})
    return records, misaligned


# --------------------------------------------------------------------------------------
# Locks and candidate tracks (instance positions in the episode GPS frame)
# --------------------------------------------------------------------------------------
def nearest_instance_fields(goal_gps, inst, target_id):
    g = np.asarray(goal_gps if goal_gps is not None else [np.nan, np.nan], dtype=float)
    near = (sorted((float(np.linalg.norm(np.asarray(v['gps']) - g)), i) for i, v in inst.items())
            if g.shape == (2,) and np.all(np.isfinite(g)) else [])
    return {'nearest_instance': near[0][1] if near else None,
            'nearest_instance_dist_m': round(near[0][0], 3) if near else None,
            'nearest_is_target': bool(near and near[0][1] == target_id)}


def track_summary(boxes, target_id):
    """{track_id: GT identity of a collect-only candidate track} from its labelled boxes.

    gt_majority is the most frequent gt_instance over the track's labelled boxes; the
    vote counts are kept so that a mixed track (association error) is visible.
    """
    by_track = defaultdict(list)
    for b in boxes:
        if b.get('track_id') is not None:
            by_track[b['track_id']].append(b)
    out = {}
    for tid in sorted(by_track):
        bs = by_track[tid]
        votes = defaultdict(int)
        labels = defaultdict(int)
        for b in bs:
            if b.get('gt_instance') is not None:
                votes[b['gt_instance']] += 1
            if b.get('label'):
                labels[b['label']] += 1
        top = max(votes.items(), key=lambda kv: (kv[1], -kv[0])) if votes else None
        steps = [b['step'] for b in bs if b.get('step') is not None]
        out[str(tid)] = {
            'n_boxes': len(bs), 'first_step': min(steps) if steps else None, 'last_step': max(steps) if steps else None,
            'gt_votes': {str(k): v for k, v in sorted(votes.items())}, 'labels': dict(labels),
            'gt_majority': top[0] if top else None,
            'gt_majority_frac': round(top[1] / sum(votes.values()), 3) if top else None,
            'majority_is_target': bool(top and top[0] == target_id),
        }
    return out


def intended_lock_record(lk, boxes, tracks, inst, target_id):
    """An [Intended Lock] labelled through the box of the same track at the same step."""
    box = next((b for b in boxes if b.get('step') == lk.get('step') and b.get('track_id') is not None
                and b.get('track_id') == lk.get('track_id')), None)
    track = tracks.get(str(lk.get('track_id')))
    return {**lk, **nearest_instance_fields(lk.get('goal_gps'), inst, target_id),
            'locked_box_found': box is not None,
            'locked_label': box.get('label') if box else None,
            'locked_gt_instance': box.get('gt_instance') if box else None,
            'locked_label_source': box.get('label_source') if box else None,
            'locked_is_target': (box.get('label') == 'target') if box and box.get('label') else None,
            'track_gt_majority': track['gt_majority'] if track else None,
            'track_majority_is_target': track['majority_is_target'] if track else None}


# --------------------------------------------------------------------------------------
# Category mapping and box labelling
# --------------------------------------------------------------------------------------
def load_mpcat40_table(path):
    table = {}
    with open(path) as fh:
        for i, line in enumerate(fh):
            fields = [f.strip() for f in re.split(r'\t|\s{2,}', line.strip()) if f.strip()]
            if i == 0 or len(fields) < 4:
                continue
            mp = fields[-1].lower()
            for name in (fields[1].lower(), fields[2].lower()):
                table.setdefault(name, mp)
    return table


def goal_category_of(name, table):
    """Map an HM3D raw semantic label to one of the six goal categories (or None)."""
    n = (name or '').lower().strip()
    mp = table.get(n)
    if mp in GOAL_CATEGORIES:
        return mp
    if re.search(r'\b(tv|television|monitor)\b', n) and not re.search(r'stand|cabinet|table|unit|mount|remote|shelf', n):
        return 'tv_monitor'
    if re.search(r'\b(sofa|couch)\b', n) and not re.search(r'cushion|pillow|table|cover|chair', n):
        return 'sofa'
    if 'chair' in n and not re.search(r'cushion|pillow|cover|leg|wheel', n):
        return 'chair'
    if re.search(r'\bbed\b', n) and not re.search(r'sheet|cover|side|frame|table|spread|lamp|light|curtain', n):
        return 'bed'
    if 'toilet' in n and not re.search(r'paper|brush|cover|tank|holder|cleaner', n):
        return 'toilet'
    if 'plant' in n and not re.search(r'pot\b|stand|holder', n):
        return 'plant'
    return None


def label_point(sem, depth, center, target_id, info, category, radius=7, frame_counts=None):
    """GT label for a detection logged only by its mask centre (scene-graph kept lines).

    The window around center_px ([x, y]) is labelled like a box; box_area_frac is unknown.
    """
    cx, cy = int(center[0]), int(center[1])
    out = label_box(sem, depth, [cx - radius, cy - radius, cx + radius + 1, cy + radius + 1],
                    target_id, info, category, frame_counts=frame_counts)
    out.update(box_area_frac=None, label_source='center_px')
    return out


def label_box(sem, depth, box, target_id, info, category, frame_counts=None):
    """GT label of one detection box from the semantic frame.

    label: 'target' if the target is the dominant non-structural instance in the box,
    'same_category' if that instance is another instance of the goal category, else 'other'.
    Raw counts are kept so that downstream analysis can apply a different rule.
    gt_instance_px_in_frame (the labelled instance's pixels in the whole frame) is a
    scale proxy that is also available for centre-labelled detections.
    """
    h, w = sem.shape
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, x1 = max(0, min(x0, w - 1)), max(0, min(max(x1, x0 + 1), w))
    y0, y1 = max(0, min(y0, h - 1)), max(0, min(max(y1, y0 + 1), h))
    patch = sem[y0:y1, x0:x1]
    area = int(patch.size)
    counts = np.bincount(patch.ravel().astype(np.int64))
    ids = np.nonzero(counts)[0]
    ranked = sorted(((int(counts[i]), int(i)) for i in ids
                     if i != 0 and info.get(int(i), {}).get('name', '') not in STRUCTURAL_NAMES), reverse=True)
    top = [{'id': i, 'px': px, 'name': info.get(i, {}).get('name'), 'goal_category': info.get(i, {}).get('goal_category')}
           for px, i in ranked[:3]]
    target_px = int(counts[target_id]) if target_id is not None and target_id < len(counts) else 0
    # Largest instance of the goal category in the box (may be occluded by a larger object).
    goal_ranked = sorted(((int(counts[i]), int(i)) for i in ids if i != 0 and (
        int(i) == target_id or info.get(int(i), {}).get('goal_category') == category)), reverse=True)
    goal_inst = {'goal_instance': goal_ranked[0][1] if goal_ranked else None,
                 'goal_instance_px': goal_ranked[0][0] if goal_ranked else 0,
                 'goal_instance_frac_of_box': round(goal_ranked[0][0] / area, 4) if goal_ranked and area else 0.0}
    if top and top[0]['id'] == target_id:
        label = 'target'
    elif top and top[0]['goal_category'] == category:
        label = 'same_category'
    else:
        label = 'other'
    dpatch = depth[y0:y1, x0:x1]
    top_depth = target_depth = None
    if top:
        m = patch == top[0]['id']
        top_depth = round(float(np.median(dpatch[m])), 3) if m.any() else None
    if target_px:
        m = patch == target_id
        target_depth = round(float(np.median(dpatch[m])), 3)
    gt = top[0]['id'] if top else None
    gt_frame_px = (int(frame_counts[gt]) if frame_counts is not None and gt is not None and gt < len(frame_counts)
                   else None)
    return {
        'label': label, 'label_source': 'box', 'gt_instance': gt, 'gt_instance_px_in_frame': gt_frame_px,
        'top_instances': top, **goal_inst,
        'target_px_in_box': target_px, 'target_frac_of_box': round(target_px / area, 4) if area else 0.0,
        'box_area_frac': round(area / float(h * w), 5), 'top_depth_m': top_depth, 'target_depth_m': target_depth,
    }


# --------------------------------------------------------------------------------------
# Rendering (habitat-sim; imported lazily)
# --------------------------------------------------------------------------------------
class Renderer:
    def __init__(self, config_path, gpu, mp_table):
        import habitat
        self._habitat = habitat
        self.config = habitat.get_config(str(config_path))
        self.gpu = gpu
        self.mp_table = mp_table
        self.sim = None
        self.scene = None
        self.info = {}

    def load_scene(self, scene_path):
        if scene_path == self.scene:
            return
        cfg = self.config.clone()
        cfg.defrost()
        cfg.SIMULATOR.SCENE = scene_path
        cfg.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = self.gpu
        cfg.freeze()
        if self.sim is None:
            self.sim = self._habitat.sims.make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)
        else:
            self.sim.reconfigure(cfg.SIMULATOR)
        self.sim.reset()
        self.scene = scene_path
        self.info = {}
        for obj in self.sim.semantic_annotations().objects:
            if obj is None:
                continue
            try:
                name = obj.category.name() if obj.category is not None else ''
            except Exception:
                name = ''
            center = None
            try:
                center = [float(v) for v in obj.aabb.center]
            except Exception:
                pass
            self.info[int(obj.semantic_id)] = {
                'name': name, 'goal_category': goal_category_of(name, self.mp_table), 'center': center,
                'object_id': str(getattr(obj, 'id', '')),
            }

    def snap_height(self, position, y_prev, max_xz_shift=0.3):
        p = np.array([position[0], y_prev, position[2]], dtype=np.float32)
        s = np.asarray(self.sim.pathfinder.snap_point(p), dtype=float)
        if np.all(np.isfinite(s)) and abs(s[0] - p[0]) <= max_xz_shift and abs(s[2] - p[2]) <= max_xz_shift:
            return float(s[1])
        return float(y_prev)

    def render(self, position, rotation, tilt_deg):
        self.sim.set_agent_state(list(position), rotation, reset_sensors=True)
        agent = self.sim.get_agent(0)
        n = int(round(abs(tilt_deg) / TILT_STEP_DEG))
        for _ in range(n):
            agent.act(LOOK_DOWN if tilt_deg < 0 else LOOK_UP)
        obs = self.sim.get_sensor_observations()
        state = self.sim.get_agent_state()
        return obs, state


# --------------------------------------------------------------------------------------
# Episode processing
# --------------------------------------------------------------------------------------
def goal_attributes(ds_ep):
    """The episode's goal attributes; the dataset stores either a dict or its Python repr."""
    import ast
    raw = ds_ep.get('goal_attributes')
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = ast.literal_eval(raw)
            return value if isinstance(value, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def load_dataset_episodes(path):
    with gzip.open(path, 'rt') as fh:
        data = json.load(fh)
    return {(e['scene_id'].split('/')[-1], str(e['episode_id']), str(e['goal_object_id'])): e
            for e in data['episodes']}


def process_episode(renderer, ep, ds_ep, run_dir, args):
    from PIL import Image
    renderer.load_scene(ep['scene_path'])
    category = ds_ep['object_category']
    start_pos, start_rot = ds_ep['start_position'], ds_ep['start_rotation']
    target_id = ep['pixel_id']
    same_cat = {i for i, v in renderer.info.items() if v['goal_category'] == category}
    if target_id is not None:
        same_cat.add(int(target_id))
    boxes, misaligned = episode_box_records(ep)
    boxes_by_step = defaultdict(list)
    for b in boxes:
        boxes_by_step[b['step']].append(b)
    steps = ep['steps']
    tilts = frame_tilts(steps)
    y = float(start_pos[1])
    frames, exposure = [], []
    pose_err_m, head_err_deg = 0.0, 0.0
    img_dir = run_dir / 'frames' / f"{ep['dataset_index']:03d}"
    for t in sorted(steps):
        rec = steps[t]
        if rec.get('agent_world') is None or rec.get('heading_deg') is None:
            continue
        pos, rot = map_pose_to_world(rec['agent_world'], rec['heading_deg'], start_pos, start_rot)
        y = renderer.snap_height(pos, y)
        pos[1] = y
        has_boxes = t in boxes_by_step
        if args.steps == 'detections' and not has_boxes:
            continue
        obs, state = renderer.render(pos, rot, tilts[t])
        back_xy, back_heading = world_to_map_pose(state.position, state.rotation, start_pos, start_rot)
        pose_err_m = max(pose_err_m, float(np.hypot(back_xy[0] - rec['agent_world'][0], back_xy[1] - rec['agent_world'][1])))
        head_err_deg = max(head_err_deg, abs(angle_diff_deg(back_heading, rec['heading_deg'])))
        sem = np.asarray(obs['semantic']).squeeze()
        depth = np.asarray(obs['depth']).squeeze()
        counts = np.bincount(sem.ravel().astype(np.int64))
        rendered_px = int(counts[target_id]) if target_id is not None and target_id < len(counts) else 0
        logged_px = steps.get(t - 1, {}).get('goal_visible_px')
        if logged_px is None:
            check = 'n/a'
        elif rendered_px == logged_px:
            check = 'exact'
        elif abs(rendered_px - logged_px) <= max(args.check_tol_px, args.check_tol_frac * logged_px):
            check = 'close'
        else:
            check = 'mismatch'
        visible = {int(i): int(counts[i]) for i in same_cat if i < len(counts) and counts[i] >= args.min_exposure_px}
        exposure.append({'step': t, 'target_px': rendered_px, 'same_category_px': visible})
        frame = {'step': t, 'position': [round(float(v), 4) for v in state.position],
                 'heading_deg': rec['heading_deg'], 'tilt_deg': tilts[t],
                 'logged_target_px': logged_px, 'rendered_target_px': rendered_px, 'check': check,
                 'image': None}
        if has_boxes:
            if not args.no_images:
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = img_dir / f'{t:04d}.jpg'
                Image.fromarray(np.asarray(obs['rgb'])[..., :3]).save(img_path, quality=args.jpeg_quality)
                frame['image'] = str(img_path.relative_to(run_dir))
            for b in boxes_by_step[t]:
                b['frame_check'] = check
                if b['box'] is not None:
                    b.update(label_box(sem, depth, b['box'], target_id, renderer.info, category,
                                       frame_counts=counts))
                elif b.get('center_px') is not None:
                    b.update(label_point(sem, depth, b['center_px'], target_id, renderer.info, category,
                                         frame_counts=counts))
        frames.append(frame)
    # Associate each logged lock with the nearest same-category instance (episode GPS frame).
    inst = {}
    for i in sorted(same_cat):
        c = renderer.info.get(i, {}).get('center')
        if c is not None:
            inst[i] = {'name': renderer.info[i]['name'],
                       'gps': [round(float(v), 3) for v in world_to_gps(c, start_pos, start_rot)]}
    tgt_gps = np.asarray(inst[target_id]['gps']) if target_id in inst else None
    for i, v in inst.items():
        v['dist_to_target_m'] = (round(float(np.linalg.norm(np.asarray(v['gps']) - tgt_gps)), 3)
                                 if tgt_gps is not None else None)
    locks = [{**lk, **nearest_instance_fields(lk.get('goal_gps'), inst, target_id)} for lk in ep['locks']]
    tracks = track_summary(boxes, target_id)
    intended = [intended_lock_record(lk, boxes, tracks, inst, target_id) for lk in ep.get('intended_locks', [])]
    end = ep['end'] or {}
    attrs = goal_attributes(ds_ep)
    return {
        'dataset_index': ep['dataset_index'], 'gpu': ep['gpu'], 'key': ep['key'], 'scene_path': ep['scene_path'],
        'category': category, 'target_instance': target_id, 'goal_world_pos': ep['goal_world_pos'],
        'success': end.get('success'), 'termination': end.get('termination'), 'num_steps': end.get('num_steps'),
        'intrinsic_attributes': attrs.get('intrinsic_attributes'),
        'extrinsic_attributes': attrs.get('extrinsic_attributes'),
        'same_category_instances': {str(k): v for k, v in inst.items()},
        'locks': locks, 'unlocks': ep['unlocks'], 'frames': frames, 'exposure': exposure, 'boxes': boxes,
        'intended_locks': intended, 'collect_tracks': tracks,
        'collect_snapshots': ep.get('collect_snapshots', []), 'landmark_utility': ep.get('landmark_utility', []),
        'qa': {'misaligned_groups': misaligned, 'max_pose_err_m': round(pose_err_m, 4),
               'collect_orphan_frames': ep.get('collect_orphans', 0),
               'intended_locks': len(intended),
               'intended_locks_unmatched': sum(not lk['locked_box_found'] for lk in intended),
               'intended_locks_on_target': sum(lk['locked_is_target'] is True for lk in intended),
               'max_heading_err_deg': round(head_err_deg, 3),
               'checks': {k: sum(f['check'] == k for f in frames) for k in ('exact', 'close', 'mismatch', 'n/a')}},
    }


def _atomic_write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, ensure_ascii=False)
    os.replace(tmp, path)


def merge_run(run_dir, tag, cfg_toggles):
    """Concatenate per-episode boxes into boxes.jsonl and summarise QA for one run."""
    qa = defaultdict(int)
    n_eps = 0
    with open(run_dir / 'boxes.jsonl.tmp', 'w') as out:
        for path in sorted(run_dir.glob('ep_*.json')):
            ep = json.load(open(path))
            n_eps += 1
            for k, v in ep['qa']['checks'].items():
                qa[k] += v
            qa['misaligned_groups'] += ep['qa']['misaligned_groups']
            for k in ('collect_orphan_frames', 'intended_locks', 'intended_locks_unmatched',
                      'intended_locks_on_target'):
                qa[k] += ep['qa'].get(k, 0)
            qa['max_pose_err_mm'] = max(qa['max_pose_err_mm'], int(round(ep['qa']['max_pose_err_m'] * 1000)))
            for b in ep['boxes']:
                out.write(json.dumps({'run': tag, 'dataset_index': ep['dataset_index'], 'category': ep['category'],
                                      'success': ep['success'], **b}, ensure_ascii=False) + '\n')
    os.replace(run_dir / 'boxes.jsonl.tmp', run_dir / 'boxes.jsonl')
    summary = {'run': tag, 'episodes': n_eps, 'toggles': cfg_toggles, 'qa': dict(qa)}
    _atomic_write_json(run_dir / 'summary.json', summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help='Run timestamps, e.g. 20261031_133130_3729343')
    ap.add_argument('--out', default='data/verification_set')
    ap.add_argument('--config', default='configs/runtime/instancenav_unseen10_1000atomic_1m.yaml')
    ap.add_argument('--dataset', default='data/datasets/instancenav/val10/val10.json.gz')
    ap.add_argument('--logs-dir', default='outputs')
    ap.add_argument('--mapping', default='tools/matterport_category_mappings.tsv')
    ap.add_argument('--gpu', type=int, default=0, help='HABITAT_SIM_V0.GPU_DEVICE_ID (after CUDA_VISIBLE_DEVICES)')
    ap.add_argument('--steps', choices=('all', 'detections'), default='all',
                    help='Render every policy step (exposure stats) or only steps with detections')
    ap.add_argument('--episodes', type=int, nargs='*', help='Only these dataset indices (testing)')
    ap.add_argument('--no-images', action='store_true')
    ap.add_argument('--jpeg-quality', type=int, default=92)
    ap.add_argument('--min-exposure-px', type=int, default=50)
    ap.add_argument('--check-tol-px', type=int, default=50)
    ap.add_argument('--check-tol-frac', type=float, default=0.02)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args(argv)

    os.chdir(REPO)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    ds = load_dataset_episodes(args.dataset)
    mp_table = load_mpcat40_table(args.mapping)
    runs = {}
    for tag in args.runs:
        eps, toggles = parse_run(tag, args.logs_dir)
        if args.episodes:
            eps = {i: e for i, e in eps.items() if i in set(args.episodes)}
        runs[tag] = (eps, toggles)
        print(f'[parse] {tag}: {len(eps)} episodes', flush=True)

    tasks = sorted(((e['scene_path'], tag, i) for tag, (eps, _) in runs.items() for i, e in eps.items()))
    renderer = Renderer(args.config, args.gpu, mp_table)
    started = time.time()
    done = 0
    for scene, tag, idx in tasks:
        run_dir = out_root / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f'ep_{idx:03d}.json'
        if path.exists() and not args.overwrite:
            done += 1
            continue
        ep = runs[tag][0][idx]
        ds_ep = ds[tuple(ep['key'])]
        t0 = time.time()
        result = process_episode(renderer, ep, ds_ep, run_dir, args)
        result['run'] = tag
        _atomic_write_json(path, result)
        done += 1
        print(f"[ep] {tag} idx={idx:3d} frames={len(result['frames'])} boxes={len(result['boxes'])} "
              f"checks={result['qa']['checks']} pose_err={result['qa']['max_pose_err_m']*1000:.1f}mm "
              f"({time.time() - t0:.1f}s, {done}/{len(tasks)}, {time.time() - started:.0f}s total)", flush=True)

    summaries = {tag: merge_run(out_root / tag, tag, toggles) for tag, (_, toggles) in runs.items()}
    try:
        rev = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:
        rev = None
    manifest_path = out_root / 'manifest.json'
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {'runs': {}}
    manifest['runs'].update(summaries)
    manifest.update({'tool': 'tools/build_verification_set.py', 'git_head': rev,
                     'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
                     'args': {k: v for k, v in vars(args).items() if k != 'episodes'}})
    _atomic_write_json(manifest_path, manifest)
    for tag, s in summaries.items():
        print(f'[summary] {tag}: {json.dumps(s["qa"])}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
