"""CoIN protocol: config values, DistractorSuccess, stop rule, dataset gating, launcher defaults."""
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from instancenav_characterization_support import env

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def coin_config():
    import habitat
    return habitat.get_config('configs/coin_contextnav.yaml')


def test_config_matches_context_nav(coin_config):
    from tools.preflight_coin import protocol_problems
    assert protocol_problems(coin_config) == []


@pytest.mark.parametrize('key, value, fragment', [
    ('SIMULATOR.RGB_SENSOR.HFOV', 79, 'RGB_SENSOR.HFOV'),
    ('TASK.SUCCESS.SUCCESS_DISTANCE', 1.0, 'SUCCESS_DISTANCE'),
    ('ENVIRONMENT.MAX_EPISODE_STEPS', 1000, 'MAX_EPISODE_STEPS'),
    ('SIMULATOR.AGENT_0.SENSORS', ['RGB_SENSOR', 'DEPTH_SENSOR', 'SEMANTIC_SENSOR'], 'ground-truth semantics'),
])
def test_protocol_drift_is_reported(coin_config, key, value, fragment):
    from tools.preflight_coin import protocol_problems
    config = coin_config.clone()
    config.defrost()
    node = config
    *path, leaf = key.split('.')
    for part in path:
        node = getattr(node, part)
    setattr(node, leaf, value)
    config.freeze()
    assert any(fragment in problem for problem in protocol_problems(config))


class _Sim:
    def __init__(self, position, distances):
        self.position, self.distances = position, distances

    def get_agent_state(self):
        return SimpleNamespace(position=self.position)

    def geodesic_distance(self, a, b, episode=None):
        assert episode is None  # must not reuse the episode's cached target path
        return self.distances[tuple(b)]


@pytest.mark.parametrize('stopped, distances, expected', [
    (True, {(1, 0, 0): 0.34, (2, 0, 0): 9.0}, 1),      # < 0.25 + 0.1
    (True, {(1, 0, 0): 0.36, (2, 0, 0): np.inf}, 0),
    (True, {(1, 0, 0): np.nan, (2, 0, 0): 0.1}, 1),    # unreachable distractors are skipped
    (False, {(1, 0, 0): 0.0, (2, 0, 0): 0.0}, 0),      # only on STOP
])
def test_distractor_success(stopped, distances, expected):
    from utils.coin_measures import DistractorSuccess
    task = SimpleNamespace(_config=SimpleNamespace(SUCCESS=SimpleNamespace(SUCCESS_DISTANCE=0.25)),
                           is_stop_called=False)
    measure = DistractorSuccess(sim=_Sim([0, 0, 0], distances), config=None, task=task)
    episode = SimpleNamespace(goals=[SimpleNamespace(distractors=[[1, 0, 0], [2, 0, 0]])])
    measure.reset_metric(episode=episode, task=task)
    assert measure.get_metric() == 0
    task.is_stop_called = stopped
    measure.update_metric(episode=episode, task=task)
    assert measure.get_metric() == expected


def _traversible():
    grid = np.ones((60, 60))
    grid[5:55, 28:32] = 0  # wall with gaps at both ends (same map as the InstanceNav goldens)
    traversible = np.ones((62, 62))
    traversible[1:61, 1:61] = grid
    return traversible


def _stop(metric, start, cell, goal_found=True, radius=0.75):
    from PSG_Nav import PSG_Nav_Agent
    agent = PSG_Nav_Agent.__new__(PSG_Nav_Agent)
    agent.dilation_deg = 0
    agent.map_resolution = 5
    task = SimpleNamespace(STOP_DISTANCE=radius)
    if metric is not None:
        task.STOP_DISTANCE_METRIC = metric
    agent.config = SimpleNamespace(TASK=task)
    goal = np.zeros((60, 60))
    goal[cell] = 1
    stg, replan, stop = agent._get_stg(_traversible(), list(start), goal, goal_found)
    return stg, bool(stop)


@pytest.mark.parametrize('start, cell, expected', [
    ((10, 10), (30, 10), False),  # 20 cells = 1.00 m
    ((10, 10), (24, 10), True),   # 14 cells = 0.70 m
    ((10, 10), (25, 10), False),  # 15 cells = 0.75 m, not below
    ((30, 24), (30, 34), True),   # 0.50 m straight line through the wall
])
def test_euclidean_stop_rule(start, cell, expected):
    assert _stop('euclidean', start, cell)[1] is expected


def test_wall_separates_the_two_stop_metrics():
    assert _stop('euclidean', (30, 24), (30, 34))[1] is True
    assert _stop('fmm', (30, 24), (30, 34))[1] is False  # geodesic detour around the wall
    assert _stop(None, (30, 24), (30, 34)) == _stop('fmm', (30, 24), (30, 34))


@pytest.mark.parametrize('radius', [.25, .5, .75, 1.])
@pytest.mark.parametrize('start,goal', [((10, 10), (20, 10)), ((30, 24), (30, 34)), ((10, 10), (11, 10))])
def test_coin_explicit_fmm_matches_instancenav_default(radius, start, goal):
    assert _stop('fmm', start, goal, radius=radius) == _stop(None, start, goal, radius=radius)


def test_euclidean_rule_needs_a_found_goal():
    for metric in ('euclidean', 'fmm'):
        assert _stop(metric, (10, 10), (14, 10), goal_found=False) == _stop('fmm', (10, 10), (14, 10), goal_found=False)


def test_detection_geometry_helpers():
    import torch
    from utils.coin_geometry import CoinGeometry
    from utils.object_detection import ObjectDetectionManager
    manager = ObjectDetectionManager.__new__(ObjectDetectionManager)
    manager.agent = SimpleNamespace(depth=np.zeros((480, 640, 1)))
    for x in (0, 100, 320, 639):
        assert manager._pixel_direction(x) == (x - 320) * 79 / 640
    assert manager._pixel_direction(torch.tensor(500)) == (torch.tensor(500) - 320) * 79 / 640
    assert manager._depth_search_bounds() == (479, 639)
    geometry = CoinGeometry((512, 512), (512, 512), 42, 79, 0.0, 5.0, 1.31)
    manager.agent = SimpleNamespace(depth=np.zeros((512, 512, 1)), coin_geometry=geometry)
    assert manager._pixel_direction(511) == pytest.approx(geometry.pixel_direction(511))
    assert manager._depth_search_bounds() == (511, 511)


def test_coin_dataset_gates():
    from configs.dataset_categories import COIN_SPLIT_CATEGORIES, dataset_to_unified_name, get_categories
    from configs.dataset_registry import get_dataset_config_path, get_default_dataset_split, is_instance_dataset
    assert is_instance_dataset('coin') and is_instance_dataset('instancenav') and not is_instance_dataset('hm3d_ovon')
    assert get_dataset_config_path('coin') == 'configs/coin_contextnav.yaml'
    assert get_default_dataset_split('coin') == 'val_unseen'
    # CoIN's categories are its split's goal categories; HM3D-OVON keeps the episode goal.
    assert get_categories('coin', 'window glass', split='val_unseen') == COIN_SPLIT_CATEGORIES['val_unseen']
    assert get_categories('hm3d_ovon', 'window glass') == ['window glass']
    with pytest.raises(ValueError):
        get_categories('coin', 'window glass')
    assert dataset_to_unified_name('hanging clothes', 'coin') == 'clothes'
    assert dataset_to_unified_name('hanging clothes', 'hm3d_ovon') == 'hanging clothes'


def test_relative_evidence_is_rejected_for_coin():
    from configs.navigation_config import validate_relative_evidence
    with env(ENABLE_RELATIVE_EVIDENCE='true', ENABLE_INSTANCE_CONFIRMATION='true'):
        with pytest.raises(ValueError, match='dataset=instancenav'):
            validate_relative_evidence(rag_mode=0, dataset='coin')
    with env():
        assert validate_relative_evidence(rag_mode=0, dataset='coin') is None


def _defaults(script):
    pattern = re.compile(r'^: "\$\{([A-Z_]+):=(.*)\}"', re.M)
    values = dict(pattern.findall((ROOT / script).read_text()))
    # run_instance100.sh keeps an old variable as a fallback: ${NEW:=${OLD:-true}}.
    return {k: re.sub(r'^\$\{[A-Z_]+:-(.*)\}$', r'\1', v) for k, v in values.items()}


# CoIN-only settings (split, subset, seed) have no InstanceNav counterpart.
COIN_ONLY = {'COIN_SPLIT', 'COIN_SUBSET', 'COIN_SEED'}
# Shared policy defaults are aligned; official benchmark geometry remains separate.
INTENDED_DIFFERENCES = {}


@pytest.mark.parametrize('radius', [0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize('metric', ['fmm', 'euclidean'])
def test_policy_radius_is_separate_from_protocol(coin_config, radius, metric):
    from tools.preflight_coin import policy_problems, protocol_problems
    cfg = coin_config.clone()
    cfg.defrost()
    cfg.TASK.STOP_DISTANCE = radius
    cfg.TASK.STOP_DISTANCE_METRIC = metric
    cfg.freeze()
    assert policy_problems(cfg) == protocol_problems(cfg) == []
    assert cfg.TASK.SUCCESS.SUCCESS_DISTANCE == 0.25


@pytest.mark.parametrize('metric,radius', [('unsupported', 0.75), ('fmm', 0), ('fmm', -1),
                                        ('fmm', float('nan')), ('fmm', float('inf'))])
def test_invalid_policy_stop_rejected(coin_config, metric, radius):
    from tools.preflight_coin import policy_problems
    cfg = coin_config.clone()
    cfg.defrost()
    cfg.TASK.STOP_DISTANCE_METRIC = metric
    cfg.TASK.STOP_DISTANCE = radius
    assert policy_problems(cfg)


def test_launcher_switch_defaults_match_run_instance100():
    coin, instance = _defaults('scripts/run_coin.sh'), _defaults('scripts/run_instance100.sh')
    shared = set(coin) - COIN_ONLY
    assert shared, 'no defaults parsed'
    assert shared <= set(instance), sorted(shared - set(instance))
    for name, (coin_value, instance_value) in INTENDED_DIFFERENCES.items():
        assert (coin[name], instance[name]) == (coin_value, instance_value), name
    same = shared - set(INTENDED_DIFFERENCES)
    assert {k: coin[k] for k in same} == {k: instance[k] for k in same}
    assert coin['ENABLE_RELATIVE_EVIDENCE'] == 'false'


def test_subset_selection_prefers_distinct_targets():
    from tools.coin_subset import select
    episodes = [SimpleNamespace(scene_id=f's{t % 2}', object_instance_id=f'obj_{t}', episode_id=str(10 * t + k))
                for t in range(4) for k in range(3)]
    keys = select(episodes, 4, seed=5)
    assert keys == select(episodes, 4, seed=5)
    chosen = {tuple(key) for key in keys}
    targets = {(ep.scene_id, ep.object_instance_id) for ep in episodes if (ep.scene_id, ep.episode_id) in chosen}
    assert len(targets) == 4
    assert len(select(episodes, 6, seed=5)) == 6
    with pytest.raises(ValueError):
        select(episodes, 13, seed=5)


def test_summary_clusters_by_target(tmp_path):
    from tools.summarize_coin import summarize
    rows = [{'scene_id': 'a/s.glb', 'episode_id': i, 'goal_object_id': i // 2, 'goal': 'x',
             'success': float(i < 2), 'spl': 0.5 * (i < 2), 'softspl': 0.3, 'distance_to_goal': 1.0,
             'distractor_success': float(i == 5), 'num_steps': 120 if i < 3 else 500} for i in range(6)]
    (tmp_path / 'experiment_0').mkdir()
    (tmp_path / 'experiment_0' / 'results.txt').write_text('\n'.join(map(str, rows)) + '\n')
    (tmp_path / 'run_info.json').write_text('{"split": "val_seen", "total": 6}')
    summary = summarize(tmp_path)
    assert summary['episodes'] == 6 and summary['targets'] == 3
    assert summary['success'] == pytest.approx(1 / 3, abs=1e-4)
    assert summary['distractor_success'] == pytest.approx(1 / 6, abs=1e-4)
    assert summary['ended_before_step_limit'] == 3
    low, high = summary['success_ci95_target_bootstrap']
    assert 0.0 <= low <= 1 / 3 <= high <= 1.0
    assert (tmp_path / 'per_episode.csv').read_text().count('\n') == 7
