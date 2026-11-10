"""Keep calibration grouping, evidence labels and parameter promotion honest."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.summarize_coin_calibration import bootstrap_interval, select_grouped, summarize
from tools.coin_calibration_nodes import caption_matches


def test_recall_counts_visible_objects_even_when_detection_is_empty():
    from tools.coin_calibration_nodes import NodeReplay
    replay = NodeReplay.__new__(NodeReplay)
    replay.graph = SimpleNamespace(set_observations=lambda obs: None,
        segment2d=lambda: None, segment2d_results=[], objects=[])
    replay.renderer = SimpleNamespace(info={1: {'name': 'chair'}})
    replay.visible, replay.history = set(), []
    replay.frame({'policy_step': 1}, {'semantic': np.ones((5, 5), dtype=int)}, 0, None)
    assert replay.visible == {1}


def test_reused_segmentation_does_not_collapse_distinct_spatial_nodes():
    from tools.coin_calibration_nodes import NodeReplay
    replay = NodeReplay.__new__(NodeReplay)
    objects = [{'image_idx': [0], 'mask_idx': [0],
                'offline_detection_origins': [(step, 0)], 'offline_gt_instances': [identity],
                'pcd': SimpleNamespace(points=np.array([[0., 0, 0], [.5, .1, .1]])),
                'detection_captions': ['chair'], 'num_detections': 1}
               for step, identity in [(1, 1), (2, -1)]]
    replay.graph = SimpleNamespace(set_observations=lambda obs: None,
        segment2d=lambda: None, segment2d_results=[], objects=objects,
        cfg=SimpleNamespace(obj_min_points=2), node_space='chair')
    replay.renderer = SimpleNamespace(info={1: {'name': 'chair'}})
    replay.ep = SimpleNamespace(goal_object_id=1, object_category='chair')
    replay.visible, replay.history = set(), []
    replay.eligible = {k: {} for k in range(1, 6)}
    replay.target_first = {k: None for k in range(1, 6)}
    replay.book_rescued = {k: set() for k in range(1, 6)}
    replay.replay_seed = 0
    replay.frame({'policy_step': 3}, {'semantic': np.ones((5, 5), dtype=int)}, 2, None)
    metrics = replay.finish()['node_metrics']['1']
    assert metrics['true_nodes'] == 1 and metrics['false_nodes'] == 1
    assert metrics['precision'] == .5


def test_grouping_never_splits_repeated_target_and_repeats_do_not_change_score():
    rows = [{'target': f't{i}', 'scores': {1: .8, 3: .5}} for i in range(12)]
    score = lambda r, k: r['scores'][k]
    first = select_grouped(rows, [1, 3], 3, score)
    repeated = select_grouped(rows + [rows[0]] * 90, [1, 3], 3, score)
    assert repeated == first
    assert first['selected'] == 1 and first['oof_delta_ci95'][0] > 0
    assert len({x['target'] for x in first['fold_assignment']}) == len(rows)


def test_no_evidence_or_tie_keeps_baseline():
    rows = [{'target': f't{i}'} for i in range(12)]
    result = select_grouped(rows, [.25, .5, .75, 1.], .75, lambda r, k: 0.)
    assert result['selected'] == .75 and not result['accepted']
    assert bootstrap_interval([]) == [0., 0.]


def test_stop_tie_breaker_uses_wrong_stops_then_actions():
    rows = [{'target': f't{i}'} for i in range(10)]
    r = select_grouped(rows, [.25, .5, .75], .75,
                       lambda r, k: float(k != .75),
                       tie=lambda r, k: (-int(k == .25), -10))
    assert r['selected'] == .5


def test_incomplete_replay_never_promotes_fitted_parameters(tmp_path):
    (tmp_path / 'audit').mkdir()
    (tmp_path / 'nodes').mkdir()
    for i in range(10):
        row = {'key': f'k{i}', 'valid': True, 'target': f't{i}', 'category': 'picture',
               'run': 'r1', 'node_schema': 6,
               'node_metrics': {str(k): {'f1': 1. if k == 1 else 0., 'target_first_atomic_step': None,
                                       'precision': 1., 'recall': 1. if k == 1 else 0.,
                                       'true_nodes': 1, 'false_nodes': 0, 'unknown_nodes': 0,
                                       'book_nodes_rescued': 0}
                                for k in range(1, 6)}}
        for stage in ('audit', 'nodes'):
            (tmp_path / stage / f'{i}.json').write_text(json.dumps(row))
    result = summarize(tmp_path)
    assert not result['complete']
    assert result['node_selection']['candidate'] == 1
    assert result['recommended'] == {'node_min_detections': 3, 'node_overrides': {},
                                     'stop_metric': 'fmm', 'stop_distance_m': .75}


@pytest.mark.parametrize('rgb_nonidentical,depth_error', [(1, 0.), (0, 1e-6)])
def test_nonexact_restored_observations_are_excluded_from_stop_fitting(
        tmp_path, rgb_nonidentical, depth_error):
    for stage in ('audit', 'stops'):
        (tmp_path / stage).mkdir()
    for key, exact in [('exact', True), ('different', False)]:
        branches = [{'frame_id': key, 'navigation_track_id': 0, 'metric': metric,
                     'radius_m': radius, 'reason': 'stop', 'extra_actions': 2, 'collisions': 0,
                     'success': radius == (.75 if exact else .25), 'stop_dtg_m': .1,
                     'target_label': {'label': 'target'}}
                    for metric, radius in [('fmm', .25), ('fmm', .5), ('fmm', .75),
                                           ('fmm', 1.), ('euclidean', .75)]]
        row = {'key': key, 'valid': True, 'target': 't', 'run': 'r1', 'stop_schema': 2,
               'branches': branches, 'excluded_locks': [],
               'rgb_nonidentical': 0 if exact else rgb_nonidentical,
               'max_depth_p99_error_m': 0 if exact else depth_error, 'max_rgb_mae': 0.}
        for stage in ('audit', 'stops'):
            (tmp_path / stage / f'{key}.json').write_text(json.dumps(row))
    result = summarize(tmp_path)
    assert result['coverage']['conditional_locks'] == 1
    assert result['coverage']['observation_excluded_locks'] == 1
    assert result['stop_selection']['target_macro_score']['0.75'] == 1.
    assert result['stop_selection']['target_macro_score']['0.25'] == 0.
    assert result['stop_totals']['fmm_0.25']['branches'] == 1


@pytest.mark.parametrize('caption,annotation,expected', [
    ('drawers', 'dresser', True), ('tv', 'tv', True), ('clothes', 'hanging clothes', True),
    ('chair', 'armchair', True), ('counter table', 'table', True),
    ('picture', 'chair', False), ('book', 'wall', False),
])
def test_real_object_with_wrong_caption_is_not_a_true_node(caption, annotation, expected):
    assert caption_matches(caption, annotation) is expected
