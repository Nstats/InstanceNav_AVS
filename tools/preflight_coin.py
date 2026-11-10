#!/usr/bin/env python3
"""Check a CoIN run's inputs before loading detectors or spending API calls.

Blocks the run when the Habitat config drifts from CoIN's official protocol
(PROTOCOL below), when the dataset or scenes are missing, when the policy would
receive ground-truth semantics, or when relative evidence is switched on (its
calibration was fit on InstanceNav only).
"""
import argparse
import importlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'habitat-lab'), str(ROOT / 'GLIP')]

# Official CoIN config, with inherited Habitat settings (depth HFOV 79,
# forward 0.25 m). Internal policy stopping is validated separately.
PROTOCOL = {
    'ENVIRONMENT.MAX_EPISODE_STEPS': 500,
    'ENVIRONMENT.ATOMIC_ACTIONS': True,
    'SIMULATOR.TURN_ANGLE': 30,
    'SIMULATOR.TILT_ANGLE': 30,
    'SIMULATOR.FORWARD_STEP_SIZE': 0.25,
    'SIMULATOR.AGENT_0.HEIGHT': 1.41,
    'SIMULATOR.AGENT_0.RADIUS': 0.17,
    'SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING': True,
    'SIMULATOR.RGB_SENSOR.WIDTH': 512,
    'SIMULATOR.RGB_SENSOR.HEIGHT': 512,
    'SIMULATOR.RGB_SENSOR.HFOV': 42,
    'SIMULATOR.RGB_SENSOR.POSITION': [0, 1.31, 0],
    'SIMULATOR.DEPTH_SENSOR.WIDTH': 512,
    'SIMULATOR.DEPTH_SENSOR.HEIGHT': 512,
    'SIMULATOR.DEPTH_SENSOR.HFOV': 79,
    'SIMULATOR.DEPTH_SENSOR.MIN_DEPTH': 0.0,
    'SIMULATOR.DEPTH_SENSOR.MAX_DEPTH': 5.0,
    'SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH': False,
    'SIMULATOR.DEPTH_SENSOR.POSITION': [0, 1.31, 0],
    'TASK.SUCCESS.SUCCESS_DISTANCE': 0.25,
    'TASK.DISTANCE_TO_GOAL.DISTANCE_TO': 'VIEW_POINTS',
    'DATASET.TYPE': 'CoINInstanceNav-v1',
}
REQUIRED_MEASUREMENTS = ('DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'SOFT_SPL', 'DISTRACTOR_SUCCESS')
POLICY_SENSORS = ['RGB_SENSOR', 'DEPTH_SENSOR']
SPLIT_SIZES = {'val_seen': 831, 'val_seen_synonyms': 359, 'val_unseen': 459}


def _lookup(config, dotted):
    node = config
    for part in dotted.split('.'):
        node = getattr(node, part)
    return node


def _same(actual, expected):
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(actual) == expected and type(actual) is type(expected)
    if isinstance(expected, (int, float)):
        return isinstance(actual, (int, float)) and abs(float(actual) - float(expected)) < 1e-9
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same(a, e) for a, e in zip(actual, expected))
    return str(actual) == str(expected)


def protocol_problems(config):
    """Differences between a loaded Habitat config and the CoIN protocol."""
    problems = []
    for key, expected in PROTOCOL.items():
        try:
            actual = _lookup(config, key)
        except AttributeError:
            problems.append(f'{key} missing (expected {expected!r})')
            continue
        if not _same(list(actual) if isinstance(expected, list) else actual, expected):
            problems.append(f'{key}={actual!r}, expected {expected!r}')
    missing = [m for m in REQUIRED_MEASUREMENTS if m not in config.TASK.MEASUREMENTS]
    if missing:
        problems.append(f'TASK.MEASUREMENTS lacks {missing}')
    if list(config.SIMULATOR.AGENT_0.SENSORS) != POLICY_SENSORS:
        problems.append(f'AGENT_0.SENSORS={list(config.SIMULATOR.AGENT_0.SENSORS)}; the policy must get only '
                        f'{POLICY_SENSORS} (no ground-truth semantics)')
    try:
        from utils.coin_geometry import CoinGeometry
        CoinGeometry.from_config(config)
    except ValueError as exc:
        problems.append(f'CoIN geometry: {exc}')
    return problems


def policy_problems(config):
    """Policy parameters are not official benchmark sensor/success constraints."""
    problems = []
    task = config.TASK
    if str(getattr(task, 'STOP_DISTANCE_METRIC', '')).lower() not in ('fmm', 'euclidean'):
        problems.append('TASK.STOP_DISTANCE_METRIC must be fmm or euclidean')
    radius = getattr(task, 'STOP_DISTANCE', None)
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
        problems.append('TASK.STOP_DISTANCE must be a finite positive distance in metres')
    return problems


def dataset_problems(config):
    import habitat
    import utils.coin_dataset  # noqa: F401
    import utils.coin_measures  # noqa: F401
    problems = []
    dataset = habitat.make_dataset(config.DATASET.TYPE, config=config.DATASET)
    episodes = dataset.episodes
    scenes = sorted({ep.scene_id for ep in episodes})
    missing = [s for s in scenes if not Path(s).is_file()]
    if missing:
        problems.append(f'Missing {len(missing)}/{len(scenes)} scene meshes; first: {missing[0]}')
    configs = sorted({ep.scene_dataset_config for ep in episodes})
    problems += [f'Missing scene dataset config: {c}' for c in configs if not Path(c).is_file()]
    subset = bool(getattr(config.DATASET, 'SUBSET_MANIFEST', ''))
    expected = SPLIT_SIZES.get(config.DATASET.SPLIT)
    if not subset and int(getattr(config.DATASET, 'SPLIT_L', -1)) < 0 and expected is not None and len(episodes) != expected:
        problems.append(f'{config.DATASET.SPLIT}: {len(episodes)} episodes, expected {expected}')
    targets = {(ep.scene_id, ep.object_instance_id) for ep in episodes}
    problems += vocabulary_problems(config.DATASET.SPLIT, episodes)
    print(f'Split={config.DATASET.SPLIT}; episodes={len(episodes)}; targets={len(targets)}; scenes={len(scenes)}; '
          f'subset={config.DATASET.SUBSET_MANIFEST if subset else "none"}')
    return problems, len(episodes)


def vocabulary_problems(split, episodes):
    """The split needs a category list (the GLIP base prompt) holding every episode goal."""
    from configs.dataset_categories import COIN_SPLIT_CATEGORIES
    if split not in COIN_SPLIT_CATEGORIES:
        return [f'{split}: no entry in COIN_SPLIT_CATEGORIES (configs/dataset_categories.py)']
    unknown = sorted({ep.object_category for ep in episodes} - set(COIN_SPLIT_CATEGORIES[split]))
    if unknown:
        return [f'{split}: episode goal categories missing from COIN_SPLIT_CATEGORIES: {unknown}']
    return []


def environment_problems(check_api=False):
    problems = []
    from configs.navigation_config import validate_relative_evidence
    try:
        validate_relative_evidence(dataset='coin')
    except ValueError as exc:
        problems.append(f'{exc} (its calibration was fit on InstanceNav; keep ENABLE_RELATIVE_EVIDENCE=false)')
    for path in ('GLIP/MODEL/glip_large_model.pth', 'data/models/sam_vit_h_4b8939.pth',
                 'data/models/groundingdino_swint_ogc.pth'):
        if not (ROOT / path).is_file():
            problems.append(f'Missing model weights: {path}')
    for name in ('habitat', 'habitat_sim', 'torch', 'skimage', 'skfmm', 'faiss', 'pytorch3d.ops',
                 'maskrcnn_benchmark.engine.predictor_glip'):
        try:
            importlib.import_module(name)
        except Exception as exc:
            problems.append(f'Import {name}: {type(exc).__name__}: {exc}')
    from utils.model_client import ModelRequestError, get_model_client, load_environment
    load_environment()
    try:
        client = get_model_client()
        print(f'Model backend={client.backend}; model={client.model}')
        if check_api:
            from PIL import Image
            answer = client.complete('What solid color is this image? Return one English word.',
                                     Image.new('RGB', (32, 32), 'red'))
            if answer.lower().strip(' .') != 'red':
                problems.append('Unexpected API vision probe answer')
            else:
                print('Vision API probe passed')
    except (ValueError, ModelRequestError) as exc:
        problems.append(str(exc))
    return problems


def load_config(config_path, split=None):
    import habitat
    config = habitat.get_config(config_path)
    if split:
        # Same override PSG_Nav applies through --dataset_split.
        config.defrost()
        config.DATASET.SPLIT = split
        config.freeze()
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/coin_contextnav.yaml')
    parser.add_argument('--split', default=None, choices=tuple(SPLIT_SIZES))
    parser.add_argument('--check-api', action='store_true')
    parser.add_argument('--count-only', action='store_true', help='Print the selected episode count and exit')
    args = parser.parse_args()
    config = load_config(args.config, args.split)
    if args.count_only:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            _, count = dataset_problems(config)
        print(count)
        return 0
    problems = [f'protocol: {p}' for p in protocol_problems(config)]
    problems += [f'policy: {p}' for p in policy_problems(config)]
    data_problems, _ = dataset_problems(config)
    problems += data_problems + environment_problems(args.check_api)
    print('Protocol: ' + json.dumps({k: _lookup(config, k) if not isinstance(v, list) else list(_lookup(config, k))
                                     for k, v in PROTOCOL.items()}, default=str))
    from configs.dataset_categories import coin_policy_categories
    print('Policy: ' + json.dumps({**coin_policy_categories(),
          'stop_distance_m': config.TASK.STOP_DISTANCE,
          'stop_distance_metric': config.TASK.STOP_DISTANCE_METRIC}))
    for problem in problems:
        print('BLOCKED:', problem)
    if not problems:
        print('CoIN preflight passed.')
    return 0 if not problems else 1


if __name__ == '__main__':
    raise SystemExit(main())
