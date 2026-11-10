"""INSTANCE_COLLECT_ONLY (实验计划.md S6): log intended locks, never lock, keep exploring."""
import json
import os
import subprocess

import numpy as np
import pytest
import torch

from configs.navigation_config import instance_collect_only_enabled
from test_instance_confirmation_controls import controls, observe_scenegraph  # noqa: F401
from test_instance_constraints import (navigation, post_navigation, Predictions,  # noqa: F401
                                       episode, observe_navigation)
from utils import collect_only


def records(capsys, tag):
    prefix = f'[{tag}] '
    return [json.loads(line[len(prefix):]) for line in capsys.readouterr().out.splitlines()
            if line.startswith(prefix)]


def all_records(text):
    out = {}
    for line in text.splitlines():
        if line.startswith('[') and '] {' in line:
            tag = line[1:line.index(']')]
            out.setdefault(tag, []).append(json.loads(line[line.index('] ') + 2:]))
    return out


def test_switch_default_and_constraints(monkeypatch):
    monkeypatch.delenv('INSTANCE_COLLECT_ONLY', raising=False)
    monkeypatch.delenv('ENABLE_CONTRADICTION_CHECK', raising=False)
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'true')
    monkeypatch.setenv('INSTANCE_CONFIRMATION_MODE', 'positive_match')
    monkeypatch.setenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', 'false')
    assert not instance_collect_only_enabled()
    monkeypatch.setenv('INSTANCE_COLLECT_ONLY', 'YES')
    assert instance_collect_only_enabled()
    monkeypatch.setenv('INSTANCE_CONFIRMATION_MODE', 'contradiction')
    with pytest.raises(ValueError, match='requires'):
        instance_collect_only_enabled()
    monkeypatch.setenv('INSTANCE_CONFIRMATION_MODE', 'positive_match')
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'false')
    with pytest.raises(ValueError, match='requires'):
        instance_collect_only_enabled()
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'true')
    monkeypatch.setenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', 'true')
    with pytest.raises(ValueError, match='cannot be combined'):
        instance_collect_only_enabled()
    monkeypatch.setenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', 'false')
    monkeypatch.setenv('INSTANCE_COLLECT_ONLY', 'sometimes')
    with pytest.raises(ValueError, match='true/false'):
        instance_collect_only_enabled()


@pytest.fixture
def collect(controls, monkeypatch, tmp_path):
    monkeypatch.setenv('INSTANCE_COLLECT_ONLY', 'true')
    monkeypatch.setenv('PSGNAV_LOG_FILE', str(tmp_path / 'gpu0.log'))
    agent, manager, observations, state = controls
    manager.reset_instance_verification(episode())
    # Maps for the snapshot helpers (cells of agent.resolution cm).
    agent.full_map = torch.zeros(1, 1, agent.map_size, agent.map_size)
    agent.fbe_free_map = torch.zeros(1, 1, agent.map_size, agent.map_size)
    agent.fbe_free_map[0, 0, 180:220, 180:220] = 1
    agent.room_map = torch.zeros(1, 9, agent.map_size, agent.map_size)
    agent.room_map[0, 0, 180:200, 180:200] = 1
    agent.agent_map_pos = (200, 200)
    agent.count_episodes = 1
    return controls


def test_navigation_lock_becomes_intended_lock_and_candidate_is_retired(collect, capsys):
    agent, manager, observations, state = collect
    observe_navigation(collect, 1)
    assert not agent.found_goal and not agent.found_possible_goal
    out = all_records(capsys.readouterr().out)
    lock, = out['Intended Lock']
    assert lock['path'] == 'navigation_confirmed' and lock['track_id'] == 0 and lock['votes'] == 1
    assert lock['candidates'][0]['retired'] is False  # state before this retirement
    assert [s['reason'] for s in out['Collect Snapshot']] == ['intended_lock']
    frame, = out['Collect Frame']
    assert frame['boxes'][0]['track_id'] == 0 and frame['boxes'][0]['confirmed']
    assert 'Goal Lock' not in out
    # The same candidate keeps being verified and voted on, but never locks again.
    observe_navigation(collect, 2)
    assert not agent.found_goal and not agent.found_possible_goal
    out = all_records(capsys.readouterr().out)
    assert 'Intended Lock' not in out and 'Goal Lock' not in out
    assert out['Collect Frame'][0]['boxes'][0]['retired'] is True
    assert len(manager._positive_evidence.tracks[0].votes) == 2
    assert any(stage == 'goal_confirmation' for stage, _ in state['stage_calls'][-2:])


def test_second_candidate_still_logs_its_own_intended_lock(collect, capsys):
    agent, manager, observations, state = collect
    observe_navigation(collect, 1)
    agent.current_obj_predictions = Predictions([[270, 220, 330, 280], [470, 220, 530, 280]])
    state['stage_answers'] = {'candidate_admission': ['yes', 'yes'], 'goal_confirmation': ['yes', 'yes']}
    capsys.readouterr()
    observe_navigation(collect, 2)
    out = all_records(capsys.readouterr().out)
    lock, = out['Intended Lock']
    assert lock['track_id'] == 1 and lock['n_retired_before'] == 1
    assert {c['track_id']: c['retired'] for c in lock['candidates']} == {0: True, 1: False}
    assert not agent.found_goal


def test_far_candidate_is_approached_but_retired_one_is_not(collect, capsys):
    agent, manager, observations, state = collect
    observe_navigation(collect, 1)
    assert manager._collect_retired == {0}
    retired_gps = manager._collect_retired_gps[0].copy()
    # The agent backs off 3 m: the same object is now a far (approach-only) box.
    base_gps = agent.get_goal_gps
    agent.get_goal_gps = lambda obs, angle, distance: base_gps(obs, angle, distance) - np.array([3., 0.])
    agent.depth.fill(6.)
    observe_navigation(collect, 2)
    assert manager._positive_evidence.tracks[-1].track_id == 0  # re-associated, retired by id
    assert not agent.found_possible_goal
    # A split track (new id) at the retired position is still not approached.
    manager._collect_retired.clear()
    manager._positive_evidence.tracks[0].node_id = 'split'  # forces a new geometric track
    observe_navigation(collect, 3)
    assert len(manager._positive_evidence.tracks) == 2
    assert not agent.found_possible_goal
    agent.get_goal_gps = base_gps
    # A far candidate elsewhere is approached as in the baseline.
    agent.current_obj_predictions = Predictions([[470, 220, 530, 280]])
    state['stage_answers'] = {'candidate_admission': ['yes'], 'goal_confirmation': ['yes']}
    observe_navigation(collect, 3)
    assert agent.found_possible_goal
    assert not np.allclose(agent.possible_goal_temp_gps, retired_gps)


def test_scenegraph_lock_becomes_intended_lock_with_kept_detection(collect, capsys):
    agent, manager, observations, state = collect
    observe_scenegraph(collect, 1)
    assert not agent.found_goal and not agent.found_possible_goal
    out = all_records(capsys.readouterr().out)
    lock, = out['Intended Lock']
    assert lock['path'] == 'scenegraph_post'
    kept = [d for d in out['Detection'] if d['kept']]
    assert len(kept) == 1 and kept[0]['center_px'] is not None and kept[0]['est_gps'] is not None
    observe_scenegraph(collect, 2)
    out = all_records(capsys.readouterr().out)
    assert 'Intended Lock' not in out and not agent.found_goal


def test_switch_off_keeps_baseline_lock(controls, monkeypatch, capsys):
    monkeypatch.setenv('INSTANCE_COLLECT_ONLY', 'false')
    agent, manager, observations, state = controls
    observe_navigation(controls, 1)
    assert agent.found_goal
    out = all_records(capsys.readouterr().out)
    assert 'Intended Lock' not in out and 'Collect Frame' not in out and 'Collect Snapshot' not in out
    assert out['Goal Lock'][0]['path'] == 'navigation_confirmed'


def test_periodic_snapshot_and_map_summaries(collect, capsys):
    agent, manager, observations, state = collect
    agent.total_steps = 20
    manager._collect_periodic_snapshot()
    manager._collect_periodic_snapshot()  # once per step
    agent.total_steps = 21
    manager._collect_periodic_snapshot()  # off cadence
    snap, = records(capsys, 'Collect Snapshot')
    assert snap['reason'] == 'periodic' and snap['step'] == 20
    assert snap['coverage'] == {'known_m2': 4.0, 'free_m2': 4.0}
    assert snap['frontiers']['frontier_cells'] > 0 and snap['frontiers']['gain_union_m2'] > 0
    assert set(snap['rooms']) == {'bedroom'}


def test_snapshot_failure_is_recorded_not_raised(collect, capsys):
    agent, manager, observations, state = collect
    del agent.room_map
    collect_only.snapshot(agent, reason='periodic')
    snap, = records(capsys, 'Collect Snapshot')
    assert 'error' in snap['rooms'] and 'known_m2' in snap['coverage']


def test_rejected_candidates_keep_passive_identity_and_raw_evidence(collect, monkeypatch, capsys):
    from PIL import Image
    agent, manager, observations, state = collect
    monkeypatch.setenv('VERIFY_BEFORE_CANDIDATE', 'false')
    identity = []
    for step, answer in enumerate(('no', 'unknown', 'yes'), 1):
        state['stage_answers']['goal_confirmation'] = [answer]
        observe_navigation(collect, step)
        out = all_records(capsys.readouterr().out)
        obs = out['Collect Observation'][0]
        decision = out['Collect Decisions'][0]['candidates'][0]
        identity.append(decision['collect_track_id'])
        assert decision['decisions'] == {'goal_confirmation': answer}
        assert decision['kept'] == (answer == 'yes')
        assert decision['navigation_track_id'] == (0 if answer == 'yes' else None)
        assert obs['candidates'][0]['detector_score'] == pytest.approx(.9)
        assert obs['candidates'][0]['distance_m'] == pytest.approx(3.)
        directory = manager._collection_recorder.directory
        assert np.array_equal(np.asarray(Image.open(directory / obs['rgb']['path'])), observations['rgb'])
        assert np.array_equal(np.load(directory / obs['depth']['path'])['depth'], agent.depth)
        request, = [r for r in out['Instance Verification'] if r['event'] == 'request']
        assert request['frame_id'] == obs['frame_id']
        assert 'A beige chair.' in request['prompt']
        assert (directory / request['image']['path']).is_file()
    assert identity == [0, 0, 0]
    assert len(manager._positive_evidence.tracks) == 1
    assert not agent.found_goal


def test_mixed_batch_and_unqueried_far_box_are_not_conflated(collect, monkeypatch, capsys):
    agent, manager, observations, state = collect
    monkeypatch.setenv('VERIFY_BEFORE_CANDIDATE', 'false')
    agent.current_obj_predictions = Predictions([[70, 220, 100, 280], [270, 220, 330, 280], [470, 220, 530, 280]])
    state['stage_answers']['goal_confirmation'] = ['no', 'unknown', 'yes']
    observe_navigation(collect, 1)
    out = all_records(capsys.readouterr().out)
    cs = out['Collect Decisions'][0]['candidates']
    assert [c['decisions']['goal_confirmation'] for c in cs] == ['no', 'unknown', 'yes']
    assert len({c['collect_track_id'] for c in cs}) == 3
    assert [c['navigation_track_id'] for c in cs] == [None, None, 0]
    agent.depth.fill(6.)
    observe_navigation(collect, 2)
    out = all_records(capsys.readouterr().out)
    assert not out['Collect Decisions'][0]['stages']
    assert all(c['decisions'] == {} for c in out['Collect Decisions'][0]['candidates'])
    assert 'Instance Verification' not in out or all(r['event'] != 'request' for r in out['Instance Verification'])


def test_collection_does_not_reuse_episode_local_ids_across_resets(collect, capsys):
    agent, manager, observations, state = collect
    observe_navigation(collect, 1)
    first = manager._collection_recorder.identity['episode_attempt_id']
    manager.reset_instance_verification(episode())
    observe_navigation(collect, 1)
    assert manager._collection_recorder.identity['episode_attempt_id'] != first
    assert len(manager._collection_recorder.store.tracks) == 1


@pytest.mark.parametrize('dataset,raw,canonical,recorded', [
    ('coin', 'armchair', 'chair', 'armchair'),
    ('coin', 'chair', 'chair', 'chair'),
    ('instancenav', 'potted_plant', 'plant', 'plant'),
])
def test_collection_category_keeps_coin_raw_identity_and_instancenav_format(
        collect, capsys, dataset, raw, canonical, recorded):
    from utils.collection_recorder import CollectionRecorder
    agent, _, _, _ = collect
    agent.dataset, agent.dataset_goal_category, agent.obj_goal = dataset, raw, canonical
    capsys.readouterr()
    CollectionRecorder(agent, episode())
    row, = all_records(capsys.readouterr().out)['Collect Episode']
    assert row['category'] == recorded
    if dataset == 'coin':
        assert row['canonical_category'] == canonical
    else:
        assert 'canonical_category' not in row


def test_identical_frames_are_content_addressed_and_cache_metadata_is_recorded(collect, monkeypatch, capsys):
    agent, manager, observations, state = collect
    monkeypatch.setenv('VERIFY_BEFORE_CANDIDATE', 'false')
    agent.scenegraph.model_client.last_request_metadata = lambda: {'cache_key': 'same-key', 'cache_hit': True}
    observe_navigation(collect, 1)
    observe_navigation(collect, 1)
    out = all_records(capsys.readouterr().out)
    first, second = out['Collect Observation']
    assert first['rgb'] == second['rgb'] and first['frame_id'] != second['frame_id']
    assert first['candidates'][0]['collect_track_id'] == second['candidates'][0]['collect_track_id']
    replies = [r for r in out['Instance Verification'] if r['event'] == 'raw_response']
    assert len(replies) == 2 and all(r['cache_hit'] and r['cache_key'] == 'same-key' for r in replies)


def test_raw_requests_survive_format_retry_and_infrastructure_failure(collect, monkeypatch, capsys):
    from utils.model_client import ModelRequestError
    agent, manager, observations, state = collect
    monkeypatch.setenv('VERIFY_BEFORE_CANDIDATE', 'false')
    calls = []
    def complete(prompt, image, **kwargs):
        calls.append(prompt)
        answer = '{broken' if len(calls) == 1 else '{"matches":["no"]}'
        kwargs['validator'](answer)
        return answer
    state['primary_complete'] = complete
    observe_navigation(collect, 1)
    out = all_records(capsys.readouterr().out)
    requests = [r for r in out['Instance Verification'] if r['event'] == 'request']
    assert [r['attempt'] for r in requests] == [1, 2]
    assert requests[0]['image'] == requests[1]['image'] and calls[0] != calls[1]
    state['primary_complete'] = lambda *a, **k: (_ for _ in ()).throw(ModelRequestError('offline'))
    with pytest.raises(ModelRequestError):
        observe_navigation(collect, 2)
    out = all_records(capsys.readouterr().out)
    assert len(out['Collect Observation']) == 1 and 'Collect Decisions' not in out
    assert any(r['event'] == 'request_error' for r in out['Instance Verification'])


def test_full_atomic_budget_with_real_act_and_exhausted_planner(monkeypatch, capsys):
    from types import SimpleNamespace
    from test_navigation_recovery import make_agent
    from utils.atomic_actions import expand_atomic_actions
    agent, state, obs = make_agent()
    # Exercise actual act(), including recovery exhaustion and prev_action bookkeeping.
    agent.detection_manager = SimpleNamespace(collect_only_enabled=lambda: True)
    state['default_action'] = 0
    atomic = 0
    while atomic < 500:
        action = agent.act(obs)
        assert action['action'] != 0 and agent.prev_action == action['action']
        for _ in expand_atomic_actions(action, enabled=True):
            atomic += 1
            if atomic == 500:
                break
    assert atomic == 500
    assert 'Collect Suppressed Stop' in capsys.readouterr().out


def test_recording_audit_and_collection_summary(collect, monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace
    from tools import check_coin_collection as audit
    from tools.summarize_coin import summarize
    agent, manager, observations, state = collect
    outdir = tmp_path / 'outputs'
    outdir.mkdir()
    logfile = outdir / 'coin_smoke_gpu0.log'
    monkeypatch.setenv('PSGNAV_LOG_FILE', str(logfile))
    monkeypatch.setenv('VERIFY_BEFORE_CANDIDATE', 'false')
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    root = tmp_path / 'data' / 'results_coin_smoke'
    (root / 'experiment_0').mkdir(parents=True)
    (root / 'run_info.json').write_text(json.dumps({'timestamp': 'smoke', 'total': 3,
        'evaluation_kind': 'collection', 'collection_schema_version': 2}))
    agent.max_episode_steps = 500
    agent.simulator = SimpleNamespace(_env=SimpleNamespace(_elapsed_steps=0))
    agent.scenegraph.model_client.last_request_metadata = lambda: {'cache_key': 'recorded-key', 'cache_hit': False}
    rows = []
    capsys.readouterr()
    for index in range(3):
        ep = SimpleNamespace(episode_id=str(index), scene_id='scene.glb')
        agent.count_episodes = index
        agent.simulator._env._elapsed_steps = 0
        manager.reset_instance_verification(ep)
        for step, answer in enumerate(['no', 'unknown', 'yes'], 1):
            state['stage_answers']['goal_confirmation'] = [answer]
            observe_navigation(collect, step)
        agent.simulator._env._elapsed_steps = 500
        manager._collection_recorder.finish_episode()
        manager._collection_recorder.finish_episode()  # duplicate update_metrics is idempotent
        rows.append(dict(scene_id='scene.glb', episode_id=str(index), goal_object_id=str(index),
                         goal='chair', num_steps=500, success=0., spl=0., softspl=.1,
                         distance_to_goal=3., distractor_success=0))
        for step in range(1, 501):
            print('[Step] ' + json.dumps({'action': 3, 'policy_step': step,
                                        'primitives': 1, 'atomic_steps': step}))
    logfile.write_text(capsys.readouterr().out)
    (root / 'experiment_0' / 'results.txt').write_text('\n'.join(map(repr, rows)) + '\n')
    report = audit.check(root)
    assert report['ok'], report['errors']
    assert report['episodes'] == 3 and report['counts']['Collect Observation'] == 9
    assert [report['counts'][answer] for answer in ('yes', 'no', 'unknown')] == [3, 3, 3]
    summary = summarize(root)
    assert summary['success'] is None and summary['success_ci95_target_bootstrap'] is None
    assert summary['completed_budget_episodes'] == 3
    asset = next((outdir / 'coin_smoke_gpu0.collect' / 'images').glob('*.png'))
    asset.unlink()
    assert not audit.check(root)['ok']


def test_three_round_command_pins_collection_and_stops_after_failure(tmp_path, monkeypatch):
    from pathlib import Path
    import shutil
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    shutil.copy('scripts/run_continuous_coin_collection.sh', scripts)
    capture = tmp_path / 'calls'
    monkeypatch.setenv('COLLECT_TEST_CALLS', str(capture))
    fake = scripts / 'run_coin.sh'
    fake.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$COLLECT_TEST_CALLS"\nexit "${COLLECT_TEST_EXIT:-0}"\n')
    command = ['bash', str(scripts / 'run_continuous_coin_collection.sh')]
    subprocess.run(command, check=True, capture_output=True)
    calls = capture.read_text().splitlines()
    assert len(calls) == 3
    for index, call in enumerate(calls, 1):
        assert '--instance-collect-only true' in call and '--subset 100' in call
        assert '--gpus 0,1' in call and call.endswith(f'_r{index}')
    assert len({call.split('--timestamp ')[1] for call in calls}) == 3
    capture.unlink()
    monkeypatch.setenv('COLLECT_TEST_EXIT', '9')
    result = subprocess.run(command, capture_output=True)
    assert result.returncode == 9 and len(capture.read_text().splitlines()) == 1


def test_launcher_exports_and_validates_collect_only(tmp_path):
    fake_python = tmp_path / 'python'
    capture = tmp_path / 'capture'
    fake_python.write_text('#!/bin/bash\nprintf "%s\\n" "$INSTANCE_COLLECT_ONLY" >> "$SWITCH_TEST_CAPTURE"\n'
                           'if [[ "$1" == "-" ]]; then cat >/dev/null; fi\n')
    fake_python.chmod(0o755)
    env = {**os.environ, 'PSGNAV_PYTHON': str(fake_python), 'SWITCH_TEST_CAPTURE': str(capture),
           'INSTANCE_CONFIRMATION_MODE': 'positive_match', 'ENABLE_INSTANCE_CONFIRMATION': 'true',
           'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK': 'false', 'INSTANCE_COLLECT_ONLY': 'false'}
    env.pop('ENABLE_CONTRADICTION_CHECK', None)
    result = subprocess.run(['bash', 'scripts/run_instance100.sh', '--check-only',
                             '--instance-collect-only', 'YES'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'INSTANCE_COLLECT_ONLY: true' in result.stdout
    assert set(capture.read_text().split()) == {'true'}
    for options in [['--instance-collect-only', 'bad'], ['--instance-collect-only'],
                    ['--instance-collect-only', 'true', '--instance-confirmation-mode', 'contradiction'],
                    ['--instance-collect-only', 'true', '--enable-instance-confirmation', 'false'],
                    ['--instance-collect-only', 'true', '--enable-post-confirmation-contradiction-check', 'true']]:
        result = subprocess.run(['bash', 'scripts/run_instance100.sh', *options], env=env,
                                capture_output=True, text=True)
        assert result.returncode == 2, options
