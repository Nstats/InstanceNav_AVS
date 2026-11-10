#!/usr/bin/env python3
"""Self-check of the CoIN evaluator and sensor geometry; no detector or VLM.

1. Annotation check (all targets, no GPU): the semantic id in each
   object_instance_id names the same category in the scene's .semantic.txt.
2. Simulator check (a seeded handful of targets per split, one GPU):
   - shapes 512x512 and the camera 1.31 m above the agent base;
   - depth resampled into the RGB frame (utils.coin_geometry) against a true
     42-degree depth render from the same pose;
   - STOP at a target view point -> success 1, DTG ~0, target pixels visible;
   - STOP at a distractor far from every view point -> success 0, distractor_success 1;
   - STOP at the start -> success equals (DTG < 0.25).

  CUDA_VISIBLE_DEVICES=2 python tools/check_coin_protocol.py --targets-per-split 2
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'habitat-lab')]
SPLITS = ('val_seen', 'val_seen_synonyms', 'val_unseen')
CONFIG = 'configs/coin_contextnav.yaml'
_ANNOTATION = re.compile(r'^(\d+),([0-9A-Fa-f]+),"(.*)",(-?\d+)$')


def annotation_check():
    from tools.coin_subset import load_episodes
    report, cache = {}, {}
    for split in SPLITS:
        targets = sorted({(ep.scene_id, ep.object_instance_id, ep.goal_object_id)
                          for ep in load_episodes(CONFIG, split)})
        bad = []
        for scene, instance, semantic_id in targets:
            path = scene.replace('.basis.glb', '.semantic.txt')
            if path not in cache:
                rows = {}
                for line in Path(path).read_text().splitlines()[1:]:
                    match = _ANNOTATION.match(line)
                    if match:
                        rows[match.group(1)] = match.group(3)
                cache[path] = rows
            name = cache[path].get(semantic_id)
            if name is None or name.lower() != instance.rsplit('_', 1)[0].lower():
                bad.append([Path(scene).name, instance, name])
        report[split] = {'targets': len(targets), 'mismatches': bad, 'ok': not bad}
    return report


def sim_config(split):
    """Policy config plus GT semantics and a true 42-degree depth render (checker only)."""
    import habitat
    config = habitat.get_config(CONFIG)
    config.defrost()
    config.DATASET.SPLIT = split
    config.DATASET.SUBSET_MANIFEST = ''
    narrow = config.SIMULATOR.DEPTH_SENSOR.clone()
    narrow.UUID = 'depth_narrow'
    narrow.HFOV = config.SIMULATOR.RGB_SENSOR.HFOV
    config.SIMULATOR.DEPTH_NARROW_SENSOR = narrow
    config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR', 'DEPTH_NARROW_SENSOR']
    config.freeze()
    return config


def depth_alignment(geometry, wide, narrow):
    aligned = geometry.align_to_rgb(geometry.sanitize(wide))[..., 0]
    truth = geometry.sanitize(narrow)[..., 0]
    valid_a, valid_t = aligned < 100, truth < 100
    both = valid_a & valid_t
    error = np.abs(aligned[both] - truth[both])
    result = {'valid_px': int(both.sum()), 'mask_agreement': round(float((valid_a == valid_t).mean()), 4)}
    if error.size:
        result.update(median_m=round(float(np.median(error)), 4), p90_m=round(float(np.percentile(error, 90)), 4),
                      p99_m=round(float(np.percentile(error, 99)), 4))
        result['ok'] = (result['median_m'] < 0.01 and result['p90_m'] < 0.05
                        and result['mask_agreement'] > 0.98)
    else:
        result['ok'] = None  # nothing valid in view (e.g. facing open space beyond 5 m)
    return result


def stop_metrics(env):
    env.step(0)  # STOP
    metrics = env.get_metrics()
    return {k: round(float(metrics[k]), 4) for k in ('success', 'distance_to_goal', 'spl', 'distractor_success')}


def pick_distractor(sim, goal):
    """A navigable distractor point at least 0.5 m (geodesic) from every view point."""
    view_points = [vp.agent_state.position for vp in goal.view_points]
    for raw in goal.distractors:
        snapped = np.asarray(sim.pathfinder.snap_point(np.asarray(raw, dtype=np.float32)))
        if not np.all(np.isfinite(snapped)):
            continue
        distance = sim.geodesic_distance(snapped, view_points)
        if np.isfinite(distance) and distance > 0.5:
            return [float(v) for v in snapped], float(distance)
    return None, None


def pixel_id(sim, goal_object_id):
    """Semantic-sensor value of the target, resolved like benchmark._GoalVisibilityTracker."""
    for obj in sim.semantic_annotations().objects:
        if getattr(obj, 'semantic_id', None) == int(goal_object_id):
            try:
                name = obj.category.name()
            except Exception:
                name = None
            return int(obj.semantic_id), name
    return int(goal_object_id), None


def check_episode(env, geometry, episode):
    sim = env.sim
    env.current_episode = episode
    obs = env.reset()
    goal = episode.goals[0]
    result = {'episode_id': episode.episode_id, 'scene': Path(episode.scene_id).name,
              'target': episode.object_instance_id, 'category': episode.object_category}
    shapes = {k: list(obs[k].shape) for k in ('rgb', 'depth', 'semantic', 'depth_narrow')}
    state = sim.get_agent_state()
    sensor = state.sensor_states.get('depth') or next(iter(state.sensor_states.values()))
    camera_height = float(sensor.position[1] - state.position[1])
    result['shapes'] = shapes
    result['camera_height_m'] = round(camera_height, 4)
    result['shapes_ok'] = (shapes['rgb'] == [512, 512, 3] and shapes['depth'] == [512, 512, 1]
                           and shapes['semantic'][:2] == [512, 512])
    result['camera_height_ok'] = abs(camera_height - 1.31) < 0.01
    result['depth_at_start'] = depth_alignment(geometry, obs['depth'], obs['depth_narrow'])
    target_px, target_name = pixel_id(sim, episode.goal_object_id)
    result['target_annotation'] = target_name

    view = goal.view_points[0].agent_state
    view_obs = sim.get_observations_at(view.position, view.rotation, keep_agent_at_new_pose=True)
    result['depth_at_view_point'] = depth_alignment(geometry, view_obs['depth'], view_obs['depth_narrow'])
    # CoIN's stored view-point rotation need not face the target, and a level 42-degree
    # camera at 1.31 m misses low objects close by; so visibility is the best of the
    # stored rotation, 12 yaws, and one LOOK_DOWN (index 5) at the stored rotation.
    import quaternion
    visible = {'stored': int((np.asarray(view_obs['semantic']) == target_px).sum())}
    yaw_px = []
    for yaw in range(0, 360, 30):
        rot = quaternion.from_rotation_vector([0.0, np.deg2rad(yaw), 0.0])
        yaw_obs = sim.get_observations_at(view.position, [rot.x, rot.y, rot.z, rot.w])
        yaw_px.append(int((np.asarray(yaw_obs['semantic']) == target_px).sum()))
    visible['best_yaw'] = max(yaw_px)
    down_obs = env.step(5)
    visible['look_down'] = int((np.asarray(down_obs['semantic']) == target_px).sum())
    result['depth_looking_down'] = depth_alignment(geometry, down_obs['depth'], down_obs['depth_narrow'])
    result['target_px_at_view_point'] = visible
    result['stop_at_view_point'] = stop_metrics(env)
    result['view_point_ok'] = (result['stop_at_view_point']['success'] == 1.0
                               and result['stop_at_view_point']['distance_to_goal'] < 0.05)

    env.current_episode = episode
    env.reset()
    point, distance = pick_distractor(sim, goal)
    result['distractor_point'] = point
    result['distractor_to_view_points_m'] = None if distance is None else round(distance, 3)
    if point is None:
        result['distractor_ok'] = None
    else:
        sim.set_agent_state(point, episode.start_rotation)
        result['stop_at_distractor'] = stop_metrics(env)
        result['distractor_ok'] = (result['stop_at_distractor']['success'] == 0.0
                                   and result['stop_at_distractor']['distractor_success'] == 1.0)

    env.current_episode = episode
    env.reset()
    start_dtg = float(env.get_metrics()['distance_to_goal'])
    result['stop_at_start'] = stop_metrics(env)
    result['start_ok'] = result['stop_at_start']['success'] == float(start_dtg < 0.25)
    return result


CHECKS = ('shapes_ok', 'camera_height_ok', 'view_point_ok', 'distractor_ok', 'start_ok')


def failures(result):
    failed = [name for name in CHECKS if result.get(name) is False]
    failed += [name for name in ('depth_at_start', 'depth_at_view_point', 'depth_looking_down')
               if result[name].get('ok') is False]
    if max(result['target_px_at_view_point'].values()) == 0:
        failed.append('target_not_visible_at_view_point')
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--targets-per-split', default=2, type=int)
    parser.add_argument('--seed', default=77, type=int)
    parser.add_argument('--skip-sim', action='store_true', help='Run only the annotation check')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    import habitat
    import utils.coin_dataset  # noqa: F401
    import utils.coin_measures  # noqa: F401
    from tools.coin_subset import select
    from utils.coin_dataset import episode_key
    from utils.coin_geometry import CoinGeometry

    report = {'annotations': annotation_check(), 'episodes': []}
    problems = [f'{s}: annotation mismatches {r["mismatches"][:3]}' for s, r in report['annotations'].items() if not r['ok']]
    if not args.skip_sim:
        for split in SPLITS:
            config = sim_config(split)
            geometry = CoinGeometry.from_config(config)
            env = habitat.Env(config=config)
            try:
                by_key = {episode_key(ep): ep for ep in env.episodes}
                for key in select(env.episodes, args.targets_per_split, args.seed):
                    result = check_episode(env, geometry, by_key[tuple(key)])
                    result['split'] = split
                    result['failed'] = failures(result)
                    report['episodes'].append(result)
                    problems += [f'{split} {result["scene"]} ep{result["episode_id"]}: {name}' for name in result['failed']]
                    print(json.dumps(result, default=str), flush=True)
            finally:
                env.close()
    report['problems'] = problems
    out = Path(args.out or f'outputs/coin_protocol_check_{time.strftime("%Y%m%d_%H%M%S")}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + '\n')
    for split, r in report['annotations'].items():
        print(f'Annotations {split}: {r["targets"]} targets, {len(r["mismatches"])} mismatches')
    for problem in problems:
        print('FAILED:', problem)
    print(f'Report: {out}')
    print('CoIN protocol check passed.' if not problems else 'CoIN protocol check FAILED.')
    return 0 if not problems else 1


if __name__ == '__main__':
    raise SystemExit(main())
