import json

import numpy as np
import pytest

from tools.build_verification_set import (
    episode_box_records, frame_tilts, goal_category_of, intended_lock_record, label_box, label_point,
    map_pose_to_world, parse_log_lines, track_summary, world_to_gps, world_to_map_pose, yaw_quat, coeff_to_quat,
)


def _yaw_coeffs(theta):
    q = yaw_quat(theta)
    return [q.x, q.y, q.z, q.w]


@pytest.mark.parametrize('start_yaw', [0.0, 0.7, -2.1, np.pi])
def test_pose_round_trip_matches_habitat_sensors(start_yaw):
    rng = np.random.default_rng(0)
    start_pos = [3.2, 0.15, -7.4]
    start_rot = _yaw_coeffs(start_yaw)
    for _ in range(20):
        agent_world = [20 + rng.uniform(-8, 8), 20 + rng.uniform(-8, 8)]
        heading = rng.uniform(-179, 179)
        pos, rot = map_pose_to_world(agent_world, heading, start_pos, start_rot)
        back_xy, back_heading = world_to_map_pose(pos, rot, start_pos, start_rot)
        assert np.allclose(back_xy, agent_world, atol=1e-6)
        assert abs((back_heading - heading + 180) % 360 - 180) < 1e-6
        # The GPS of the agent position equals full_pose minus the map offset.
        gps = world_to_gps(pos, start_pos, start_rot)
        assert np.allclose(gps, [agent_world[0] - 20, 20 - agent_world[1]], atol=1e-6)


def test_start_pose_maps_to_start_state():
    start_pos, start_rot = [1.0, 0.2, 2.0], _yaw_coeffs(0.9)
    pos, rot = map_pose_to_world([20.0, 20.0], 0.0, start_pos, start_rot)
    assert np.allclose(pos, start_pos)
    q0 = coeff_to_quat(start_rot)
    assert np.allclose([rot.w, rot.x, rot.y, rot.z], [q0.w, q0.x, q0.y, q0.z])


def test_opening_scan_tilts():
    # PSG_Nav.act: 1 look down, 2-7 turn, 8 look down, 9-14 turn, 15-16 look up.
    actions = {1: 5, 8: 5, 15: 4, 16: 4}
    steps = {t: {'action': actions.get(t, 6), 'primitives': 1 if t in actions else 2} for t in range(1, 20)}
    tilts = frame_tilts(steps)
    assert tilts[1] == 0
    assert all(tilts[t] == -30 for t in range(2, 9))
    assert all(tilts[t] == -60 for t in range(9, 16))
    assert tilts[16] == -30 and tilts[17] == 0 and tilts[19] == 0


def _line(tag, body):
    return f'[{tag}] ' + json.dumps(body)


def test_parse_groups_boxes_and_stage_answers():
    lines = [
        _line('Run Header', {'toggles': {'VERIFY_BEFORE_CANDIDATE': 'true'}}),
        '[Schedule] event=claim gpu=0 dataset_index=7 episode_id=42 scene_id=data/s/A/A.basis.glb goal_object_id=99',
        _line('Goal Mapping', {'pixel_id': 99, 'goal_world_pos': [1, 0, 2]}),
        _line('Instance Verification', {'event': 'result', 'stage': 'candidate_admission', 'step': 5,
                                        'box_indices': [0, 1, 2], 'matches': ['yes', 'no', 'yes'],
                                        'verification_id': 'a'}),
        _line('Instance Verification', {'event': 'result', 'stage': 'goal_confirmation', 'step': 5,
                                        'box_indices': [2], 'matches': ['unknown'], 'verification_id': 'b'}),
        _line('Instance Verification', {'event': 'filter_result', 'stage': 'candidate_pipeline', 'step': 5,
                                        'verification_id': 'p', 'admitted_box_indices': [0, 2],
                                        'confirmation_box_indices': [2], 'confirmed_box_indices': [],
                                        'kept_box_indices': [0], 'rejected_box_indices': [1, 2],
                                        'contradicted_box_indices': [], 'checked_box_indices': [0, 1, 2]}),
        _line('Detection', {'step': 5, 'kept': False, 'box': [10, 10, 20, 20], 'source': 'navigation', 'prediction_idx': 3}),
        _line('Detection', {'step': 5, 'kept': False, 'box': [30, 30, 40, 40], 'source': 'navigation', 'prediction_idx': 4}),
        _line('Detection', {'step': 5, 'kept': True, 'box': [50, 50, 60, 60], 'source': 'navigation', 'prediction_idx': 1,
                            'est_distance_m': 6.1}),
        _line('Step', {'policy_step': 5, 'action': 1, 'primitives': 1, 'agent_world': [20, 20], 'heading_deg': 0.0,
                       'goal_visible_px': 0}),
        _line('Detection', {'step': 6, 'kept': True, 'box': [1, 1, 5, 5], 'source': 'navigation'}),
        _line('EpisodeEnd', {'success': 0.0, 'termination': 'stop_called'}),
    ]
    eps, toggles = parse_log_lines(lines, gpu=0)
    assert toggles['VERIFY_BEFORE_CANDIDATE'] == 'true'
    ep = eps[7]
    assert ep['pixel_id'] == 99 and ep['key'] == ['A.basis.glb', '42', '99']
    records, misaligned = episode_box_records(ep)
    assert misaligned == 0
    by_index = {r['box_index']: r for r in records if r['verified']}
    assert by_index[0]['kept'] and by_index[0]['box'] == [50, 50, 60, 60] and by_index[0]['admission'] == 'yes'
    assert by_index[1]['admission'] == 'no' and by_index[1]['box'] == [10, 10, 20, 20]
    assert by_index[2]['admission'] == 'yes' and by_index[2]['confirmation'] == 'unknown'
    assert by_index[2]['confirmation_called'] and by_index[2]['box'] == [30, 30, 40, 40]
    loose = [r for r in records if not r['verified']]
    assert len(loose) == 1 and loose[0]['step'] == 6


def test_skipped_kept_box_does_not_drop_the_others():
    # Boxes 0 (far, unverified) and 2 (near, confirmed yes) are kept, box 1 rejected.
    # Box 0's track was unavailable, so only box 2 is re-emitted.
    lines = [
        '[Schedule] event=claim gpu=0 dataset_index=1 episode_id=5 scene_id=data/s/B/B.basis.glb goal_object_id=3',
        _line('Instance Verification', {'event': 'result', 'stage': 'goal_confirmation', 'step': 9,
                                        'box_indices': [1, 2], 'matches': ['no', 'yes'], 'verification_id': 'c'}),
        _line('Instance Verification', {'event': 'filter_result', 'stage': 'candidate_pipeline', 'step': 9,
                                        'verification_id': 'p', 'admitted_box_indices': [0, 1, 2],
                                        'confirmation_box_indices': [1, 2], 'confirmed_box_indices': [2],
                                        'kept_box_indices': [0, 2], 'rejected_box_indices': [1],
                                        'contradicted_box_indices': [], 'checked_box_indices': [1, 2]}),
        _line('Detection', {'step': 9, 'kept': False, 'box': [0, 0, 5, 5], 'prediction_idx': 4}),
        _line('Detection', {'step': 9, 'kept': True, 'box': [7, 7, 9, 9], 'prediction_idx': 6,
                            'est_gps': [1.0, 2.0], 'est_distance_m': 2.5}),
        _line('EpisodeEnd', {'success': 1.0}),
    ]
    eps, _ = parse_log_lines(lines)
    records, misaligned = episode_box_records(eps[1])
    by_index = {r['box_index']: r for r in records}
    assert misaligned == 1
    assert by_index[1]['box'] == [0, 0, 5, 5] and by_index[1]['confirmation'] == 'no'
    assert by_index[2]['box'] == [7, 7, 9, 9] and by_index[2]['confirmation'] == 'yes'
    assert by_index[0]['box'] is None and by_index[0]['kept']


def test_goal_category_mapping_keywords():
    table = {}
    assert goal_category_of('armchair', table) == 'chair'
    assert goal_category_of('chair cushion', table) is None
    assert goal_category_of('couch', table) == 'sofa'
    assert goal_category_of('led tv', table) == 'tv_monitor'
    assert goal_category_of('tv stand', table) is None
    assert goal_category_of('bedside table', table) is None
    assert goal_category_of('bed', table) == 'bed'
    assert goal_category_of('toilet paper', table) is None
    assert goal_category_of('potted plant', table) == 'plant'


def test_label_box_rules():
    sem = np.zeros((10, 10), dtype=np.uint32)
    depth = np.full((10, 10), 2.0, dtype=np.float32)
    info = {1: {'name': 'wall'}, 5: {'name': 'chair', 'goal_category': 'chair'},
            6: {'name': 'chair', 'goal_category': 'chair'}, 7: {'name': 'table', 'goal_category': None}}
    sem[:, :] = 1
    sem[2:8, 2:8] = 5           # target chair dominates the box
    out = label_box(sem, depth, [0, 0, 10, 10], 5, info, 'chair')
    assert out['label'] == 'target' and out['target_px_in_box'] == 36 and out['target_depth_m'] == 2.0
    out = label_box(sem, depth, [0, 0, 10, 10], 6, info, 'chair')
    assert out['label'] == 'same_category' and out['gt_instance'] == 5
    sem[2:8, 2:8] = 7
    sem[8:10, 2:5] = 5          # the target chair peeks out under a larger table
    out = label_box(sem, depth, [0, 0, 10, 10], 5, info, 'chair')
    assert out['label'] == 'other'
    assert out['goal_instance'] == 5 and out['goal_instance_px'] == 6 and out['goal_instance_frac_of_box'] == 0.06


def test_label_point_uses_center_window():
    sem = np.ones((40, 40), dtype=np.uint32)
    depth = np.full((40, 40), 3.0, dtype=np.float32)
    info = {1: {'name': 'wall'}, 9: {'name': 'potted plant', 'goal_category': 'plant'}}
    sem[20:30, 5:12] = 9        # plant around x=8, y=25
    counts = np.bincount(sem.ravel().astype(np.int64))
    out = label_point(sem, depth, [8, 25], 9, info, 'plant', radius=3, frame_counts=counts)
    assert out['label'] == 'target' and out['label_source'] == 'center_px'
    assert out['box_area_frac'] is None and out['gt_instance_px_in_frame'] == 70


def _collect_only_log():
    # Collect-only frame at step 12: box 1 rejected, boxes 0 (far, no [Detection] line) and
    # 2 (near, confirmed) kept. Box 2 then reaches the vote threshold -> intended lock.
    # The scene-graph style extra kept line (centre only) must not displace anything.
    return [
        '[Schedule] event=claim gpu=1 dataset_index=4 episode_id=8 scene_id=data/s/C/C.basis.glb goal_object_id=7',
        _line('Collect Snapshot', {'episode': 0, 'step': 10, 'reason': 'periodic', 'coverage': {'known_m2': 12.0},
                                   'frontiers': {'gain_union_m2': 40.0}, 'rooms': {}, 'n_candidates': 0, 'n_retired': 0}),
        _line('Step', {'policy_step': 11, 'action': 1, 'primitives': 1, 'agent_world': [20, 20], 'heading_deg': 0.0}),
        _line('Instance Verification', {'event': 'result', 'stage': 'goal_confirmation', 'step': 12,
                                        'box_indices': [1, 2], 'matches': ['no', 'yes'], 'verification_id': 'v'}),
        _line('Collect Frame', {'episode': 0, 'step': 12, 'boxes': [
            {'box_index': 0, 'track_id': 3, 'box': [100, 10, 120, 40], 'gps': [5.0, 1.0], 'distance_m': 6.2,
             'confirmed': False, 'positive': False, 'retired': False},
            {'box_index': 2, 'track_id': 4, 'box': [200, 50, 260, 150], 'gps': [1.0, 0.5], 'distance_m': 2.1,
             'confirmed': True, 'positive': True, 'retired': False}]}),
        _line('Instance Verification', {'event': 'filter_result', 'stage': 'candidate_pipeline', 'step': 12,
                                        'verification_id': 'p', 'admitted_box_indices': [0, 1, 2],
                                        'confirmation_box_indices': [1, 2], 'confirmed_box_indices': [2],
                                        'kept_box_indices': [0, 2], 'rejected_box_indices': [1],
                                        'contradicted_box_indices': [], 'checked_box_indices': [1, 2]}),
        _line('Detection', {'step': 12, 'kept': False, 'box': [0, 0, 5, 5], 'prediction_idx': 4}),
        _line('Detection', {'step': 12, 'kept': True, 'box': [200, 50, 260, 150], 'prediction_idx': 6,
                            'source': 'navigation', 'est_gps': [1.0, 0.5], 'est_distance_m': 2.1}),
        _line('Detection', {'step': 12, 'kept': True, 'center_px': [230, 100], 'source': 'scenegraph'}),
        _line('Intended Lock', {'episode': 0, 'step': 12, 'path': 'navigation_confirmed', 'track_id': 4,
                                'goal_gps': [1.0, 0.5], 'distance_m': 2.1, 'box': [200, 50, 260, 150], 'votes': 1,
                                'n_retired_before': 0, 'candidates': []}),
        _line('Landmark Utility', {'episode': 0, 'step': 12, 'best': 0, 'landmarks': []}),
        _line('Step', {'policy_step': 12, 'action': 1, 'primitives': 1, 'agent_world': [20, 20], 'heading_deg': 0.0}),
        # A [Collect Frame] whose filter_result never comes (step mismatch) is counted, not attached.
        _line('Collect Frame', {'episode': 0, 'step': 13, 'boxes': []}),
        _line('Instance Verification', {'event': 'filter_result', 'stage': 'candidate_pipeline', 'step': 14,
                                        'verification_id': 'q', 'kept_box_indices': [], 'rejected_box_indices': []}),
        _line('EpisodeEnd', {'success': 0.0, 'termination': 'max_steps'}),
    ]


def test_collect_frame_supplies_kept_geometry_and_tracks():
    eps, _ = parse_log_lines(_collect_only_log())
    ep = eps[4]
    assert ep['collect_orphans'] == 1
    assert len(ep['intended_locks']) == 1 and 'episode' not in ep['intended_locks'][0]
    assert len(ep['collect_snapshots']) == 1 and len(ep['landmark_utility']) == 1
    assert ep['groups'][0]['collect'] is not None and ep['groups'][1]['collect'] is None
    records, misaligned = episode_box_records(ep)
    assert misaligned == 0   # the far kept box has no [Detection] line but is placed from [Collect Frame]
    by_index = {r['box_index']: r for r in records if r['verified'] and r['step'] == 12}
    assert by_index[0]['box'] == [100, 10, 120, 40] and by_index[0]['est_gps'] == [5.0, 1.0]
    assert by_index[0]['track_id'] == 3 and by_index[0]['source'] is None
    assert by_index[2]['track_id'] == 4 and by_index[2]['prediction_idx'] == 6
    assert by_index[2]['collect_confirmed'] and by_index[2]['confirmation'] == 'yes'
    assert by_index[1]['box'] == [0, 0, 5, 5] and by_index[1]['track_id'] is None


def test_baseline_logs_unchanged_by_collect_fields():
    # Without [Collect Frame] the new fields are empty and kept lines are placed by order,
    # as before. Here that order-based placement is wrong (the far box 0 has no line of its
    # own and receives box 2's), which is the case [Collect Frame] resolves.
    lines = [l for l in _collect_only_log() if not l.startswith('[Collect Frame]')]
    eps, _ = parse_log_lines(lines)
    records, misaligned = episode_box_records(eps[4])
    assert misaligned == 0 and eps[4]['collect_orphans'] == 0
    assert all(r['track_id'] is None for r in records)
    by_index = {r['box_index']: r for r in records if r['verified'] and r['step'] == 12}
    assert by_index[0]['box'] == [200, 50, 260, 150]


def test_intended_lock_is_labelled_through_its_track_box():
    boxes = [
        {'step': 12, 'track_id': 4, 'label': 'target', 'gt_instance': 7, 'label_source': 'box'},
        {'step': 15, 'track_id': 4, 'label': 'target', 'gt_instance': 7},
        {'step': 18, 'track_id': 4, 'label': 'same_category', 'gt_instance': 9},
        {'step': 12, 'track_id': 3, 'label': 'same_category', 'gt_instance': 9},
        {'step': 12, 'track_id': None, 'label': 'other', 'gt_instance': 2},
    ]
    tracks = track_summary(boxes, target_id=7)
    assert tracks['4']['gt_majority'] == 7 and tracks['4']['gt_majority_frac'] == 0.667
    assert tracks['4']['majority_is_target'] and not tracks['3']['majority_is_target']
    assert tracks['4']['first_step'] == 12 and tracks['4']['last_step'] == 18
    inst = {7: {'gps': [1.0, 0.6]}, 9: {'gps': [5.0, 1.0]}}
    rec = intended_lock_record({'step': 12, 'track_id': 4, 'goal_gps': [1.0, 0.5]}, boxes, tracks, inst, 7)
    assert rec['locked_box_found'] and rec['locked_is_target'] and rec['locked_gt_instance'] == 7
    assert rec['nearest_instance'] == 7 and rec['track_majority_is_target']
    miss = intended_lock_record({'step': 13, 'track_id': 4, 'goal_gps': None}, boxes, tracks, inst, 7)
    assert not miss['locked_box_found'] and miss['locked_is_target'] is None and miss['nearest_instance'] is None
