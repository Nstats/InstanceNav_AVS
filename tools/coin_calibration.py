#!/usr/bin/env python3
"""Offline CoIN calibration. GT is used only here, never by the navigation policy.

Replay immutable collect-only actions before fitting any parameter. Reject episodes
whose observations/poses/budget cannot be reproduced. No model API is called.
"""
import argparse
from collections import defaultdict
import hashlib
import fcntl
import gzip
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'habitat-lab'), str(ROOT / 'GLIP')]

import numpy as np

DEFAULT_RUNS = [f'coin_val_unseen_collect_20261108_181523_1477518_r{r}' for r in (1, 2)]


def read_attempts(run):
    """Link logs to schema-2 attempts by worker-local episode number."""
    for directory in sorted((ROOT / 'outputs').glob(run + '_gpu*.collect')):
        attempts = {}
        preceding_goals = []
        for line in (directory / 'events.jsonl').open():
            event = json.loads(line)
            if event['tag'] == 'Collect Episode':
                attempts[event['episode']] = {'header': event, 'frames': defaultdict(list),
                                             'locks': defaultdict(list), 'steps': [],
                                             'directory': str(directory), 'run': run,
                                             'preceding_goals': list(preceding_goals)}
                preceding_goals.append(event['category'])
            item = attempts.get(event.get('episode'))
            if item is None:
                continue
            if event['tag'] == 'Collect Observation':
                item['frames'][event['step']].append(event)
            elif event['tag'] == 'Collect Intended Lock':
                item['locks'][event['step']].append(event)
            elif event['tag'] == 'Collect End':
                item['end'] = event
        for line in directory.with_suffix('.log').open():
            if line.startswith('[Step] '):
                step = json.loads(line[len('[Step] '):])
                if step['episode'] in attempts:
                    attempts[step['episode']]['steps'].append(step)
        for item in attempts.values():
            if item.get('end', {}).get('atomic_steps') == 500:
                yield item


def attempt_key(item):
    h = item['header']
    return f"{item['run']}_{Path(h['scene_id']).stem}_{h['episode_id']}"


def write_report(dest, report):
    """Keep small metrics separate from detailed histories; commit files atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    history = report.pop('node_history', None)
    if history is not None:
        history_path = dest.with_suffix('.history.jsonl.gz')
        tmp = history_path.with_suffix(f'.{os.getpid()}.tmp')
        with gzip.open(tmp, 'wt') as stream:
            for row in history:
                stream.write(json.dumps(row) + '\n')
        tmp.replace(history_path)
        report['node_history_file'] = history_path.name
    tmp = dest.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(dest)


def observations(sim, ep, geometry):
    from tools.build_verification_set import world_to_map_pose
    # Use Habitat's sensor adapter too: it clips depth and adds its channel axis.
    obs = sim._sensor_suite.get_observations(sim.get_sensor_observations())
    state = sim.get_agent_state()
    pose, angle = world_to_map_pose(state.position, state.rotation, ep.start_position, ep.start_rotation)
    obs['gps'] = np.asarray([pose[0] - 20, 20 - pose[1]], dtype=np.float32)
    obs['compass'] = np.asarray([np.deg2rad(angle)], dtype=np.float32)
    geometry.prepare_observations(obs)
    return obs, pose, angle


def audit_frame(item, step, obs, pose, angle, atomic, report):
    from tools.build_verification_set import angle_diff_deg
    err = float(np.max(np.abs(np.asarray(pose) - step['agent_world'])))
    report['max_position_error_m'] = max(report['max_position_error_m'], err)
    report['max_heading_error_deg'] = max(report['max_heading_error_deg'], abs(angle_diff_deg(angle, step['heading_deg'])))
    if err > .002 or abs(angle_diff_deg(angle, step['heading_deg'])) > .02:
        report['pose_mismatches'] += 1
    for frame in item['frames'].get(step['policy_step'], []):
        rgb = obs['rgb'][..., :3]
        sha = hashlib.sha256(str((rgb.shape[1], rgb.shape[0])).encode() + rgb.tobytes()).hexdigest()
        report['frames_checked'] += 1
        if sha != frame['rgb']['pixel_sha256']:
            # Cross-GPU rasterization may differ by a few byte values. Record
            # the actual discrepancy, never silently treat changed views as equal.
            from PIL import Image
            original = np.asarray(Image.open(Path(item['directory']) / frame['rgb']['path']))
            diff = np.abs(rgb.astype(float) - original)
            mae = float(diff.mean())
            bad = float((diff.max(axis=2) > 8).mean())
            report['rgb_nonidentical'] += 1
            report['max_rgb_mae'] = max(report['max_rgb_mae'], mae)
            report['max_rgb_bad_fraction'] = max(report['max_rgb_bad_fraction'], bad)
            if mae > .25 or bad > .001:
                report['rgb_mismatches'] += 1
        original_depth = np.load(Path(item['directory']) / frame['depth']['path'])['depth']
        difference = np.abs(original_depth - obs['depth'])
        valid = (original_depth < 60) & (obs['depth'] < 60)
        invalid_disagreement = float(((original_depth < 60) != (obs['depth'] < 60)).mean())
        p99 = float(np.percentile(difference[valid], 99)) if valid.any() else 0.
        report['max_depth_p99_error_m'] = max(report['max_depth_p99_error_m'], p99)
        if p99 > .01 or invalid_disagreement > .001:
            report['depth_mismatches'] += 1
        if atomic != frame['atomic_steps_before']:
            report['budget_mismatches'] += 1


def replay(item, ep, renderer, geometry, on_frame=None):
    from tools.build_verification_set import coeff_to_quat
    from habitat.tasks.nav.nav import DistanceToGoal
    renderer.load_scene(ep.scene_id)
    sim = renderer.sim
    sim.reset()
    sim.set_agent_state(ep.start_position, coeff_to_quat(ep.start_rotation), reset_sensors=True)
    ep._shortest_path_cache = None
    dtg_measure = DistanceToGoal(sim, renderer.config.TASK.DISTANCE_TO_GOAL)
    dtg_measure.reset_metric(ep)
    report = {'key': attempt_key(item), 'run': item['run'], 'category': ep.object_category,
              'target': f'{ep.scene_id}:{ep.goal_object_id}', 'episode_id': ep.episode_id,
              'frames_checked': 0, 'rgb_nonidentical': 0, 'rgb_mismatches': 0,
              'depth_mismatches': 0, 'pose_mismatches': 0, 'budget_mismatches': 0,
              'max_position_error_m': 0., 'max_heading_error_deg': 0.,
              'max_depth_p99_error_m': 0., 'max_rgb_mae': 0., 'max_rgb_bad_fraction': 0.,
              'max_dtg_error_m': 0.}
    atomic = 0
    for step in item['steps']:
        obs, pose, angle = observations(sim, ep, geometry)
        audit_frame(item, step, obs, pose, angle, atomic, report)
        if on_frame:
            on_frame(step, obs, atomic, sim)
        action = step['action']
        for _ in range(step['primitives']):
            sim.step(3 if action == 6 else action)
            atomic += 1
            dtg_measure.update_metric(ep)
        dtg = float(dtg_measure.get_metric())
        if np.isfinite(dtg) and step.get('dtg_gt') is not None:
            report['max_dtg_error_m'] = max(report['max_dtg_error_m'], abs(dtg - step['dtg_gt']))
        if atomic != step['atomic_steps']:
            report['budget_mismatches'] += 1
    report['atomic_steps'] = atomic
    report['valid'] = bool(atomic == 500 and report['frames_checked'] > 0 and
        not any(report[k] for k in ('rgb_mismatches', 'depth_mismatches', 'pose_mismatches', 'budget_mismatches')) and
        report['max_dtg_error_m'] <= .002)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', default=DEFAULT_RUNS)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/coin_calibration_20261109')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--stage', choices=['audit', 'nodes', 'stops'], default='audit')
    parser.add_argument('--episode', help='Optional episode id for a replay smoke check')
    args = parser.parse_args()
    if args.stage in ('nodes', 'stops'):
        import torch
        torch.cuda.set_device(args.gpu)
        os.environ.setdefault('HF_HOME', str(ROOT / 'data/huggingface'))
        os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(ROOT / 'data/huggingface/hub'))
        os.environ.setdefault('TRANSFORMERS_CACHE', str(ROOT / 'data/huggingface/hub'))
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    from tools.coin_subset import load_episodes
    from tools.build_verification_set import Renderer
    from utils.coin_geometry import CoinGeometry
    episodes = {(e.scene_id, e.episode_id): e for e in load_episodes('configs/coin_contextnav.yaml', 'val_unseen')}
    renderer = Renderer('configs/coin_contextnav.yaml', args.gpu, {})
    renderer.config.defrost()
    renderer.config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR']
    renderer.config.freeze()
    geometry = CoinGeometry.from_config(renderer.config)
    sources = ['tools/coin_calibration.py', 'tools/coin_calibration_nodes.py',
               'tools/coin_calibration_stops.py', 'PSG_Nav.py', 'utils/map_manager.py',
               'utils/utils_fmm/mapping.py', 'utils/utils_fmm/fmm_planner.py',
               'utils/utils_sg_uq/merge_utils.py', 'configs/dataset_categories.py',
               'configs/coin_contextnav.yaml']
    provenance = {'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                     for name in sources},
                  'renderer_gpu': args.gpu, 'policy_gt_access': False}
    evaluator = None
    if args.stage == 'nodes':
        from tools.coin_calibration_nodes import NodeReplay
        evaluator = NodeReplay(renderer.config, geometry, args.out / 'perception_cache')
    elif args.stage == 'stops':
        from tools.coin_calibration_stops import StopReplay
        evaluator = StopReplay(renderer.config, geometry)
    args.out.mkdir(parents=True, exist_ok=True)
    items = [x for run in args.runs for x in read_attempts(run)]
    items.sort(key=lambda i: (i['header']['scene_id'], i['run'], i['header']['episode_id']))
    if args.episode:
        items = [i for i in items if i['header']['episode_id'] == args.episode]
    items = items[args.shard::args.shards]
    if args.limit:
        items = items[:args.limit]
    for item in items:
        key = attempt_key(item)
        dest = args.out / args.stage / f'{key}.json'
        lock_dir = args.out / '.locks'
        lock_dir.mkdir(exist_ok=True)
        lock = (lock_dir / f'{args.stage}_{key}.lock').open('w')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        if dest.exists():
            existing = json.loads(dest.read_text())
            if (existing.get('valid') and (args.stage != 'nodes' or existing.get('node_schema') == 6)
                    and (args.stage != 'stops' or existing.get('stop_schema') == 2)):
                lock.close()
                continue
        ep = episodes[(item['header']['scene_id'], item['header']['episode_id'])]
        if evaluator:
            audit_path = args.out / 'audit' / f'{key}.json'
            if not audit_path.exists() or not json.loads(audit_path.read_text())['valid']:
                print('SKIP unverified trajectory:', key, flush=True)
                lock.close()
                continue
            evaluator.begin(item, ep, renderer)
        report = replay(item, ep, renderer, geometry,
                        on_frame=evaluator.frame if evaluator else None)
        if evaluator:
            report.update(evaluator.finish())
        report['provenance'] = provenance
        write_report(dest, report)
        lock.close()
        print(json.dumps({k: v for k, v in report.items() if k not in ('node_history', 'branches')}), flush=True)
    if renderer.sim:
        renderer.sim.close()


if __name__ == '__main__':
    main()
