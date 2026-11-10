import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from configs.navigation_config import relative_evidence_run_header, validate_relative_evidence
from utils.relative_evidence import CandidatePool, Calibration, decision_snapshot
from utils.relative_comparison import build_pair_prompt, compare_candidates, marked_view
from utils.relative_evidence_navigation import RelativeEvidenceNavigation
from utils.model_client import ModelRequestError
from test_instance_constraints import navigation, Predictions, episode, observe_navigation
from test_instance_confirmation_controls import observe_scenegraph
from test_navigation_recovery import make_agent

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / 'configs/relative_evidence_s6.json'


def settings(monkeypatch):
    values = {'ENABLE_RELATIVE_EVIDENCE': 'true', 'VERIFY_BEFORE_CANDIDATE': 'false',
              'INSTANCE_CONFIRMATION_THRESHOLD': '1', 'INSTANCE_CONFIRMATION_MODE': 'positive_match',
              'ENABLE_INSTANCE_CONFIRMATION': 'true', 'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK': 'false',
              'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY': 'false', 'INSTANCE_COLLECT_ONLY': 'false',
              'RAG_MODE': '0', 'RELATIVE_EVIDENCE_CALIBRATION': str(CALIBRATION),
              'RELATIVE_EVIDENCE_USE_U': 'true', 'RELATIVE_EVIDENCE_EARLY_STOP': 'true',
              'RELATIVE_EVIDENCE_PAIRWISE': 'true', 'RELATIVE_EVIDENCE_DROP_REJECTED': 'true',
              'ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION': 'false'}
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('ENABLE_CONTRADICTION_CHECK', raising=False)


def constant_model(c=0, tau=.3):
    return {'mean': [0.] * 6, 'scale': [1.] * 6, 'coef': [0.] * 6, 'intercept': 0.,
            'params': {'alpha': .1, 'c': c, 'tau': tau}}


def add(pool, step=1, gps=(1., 0.), answer='yes', **kw):
    return pool.add(event_id=kw.pop('event_id', (step, len(pool.seen_events))), step=step, gps=gps,
                    answer=answer, agent_position=(0., 0.), **kw)


def test_streaming_seed_clusters_unknown_and_sighting_boundaries():
    pool = CandidatePool()
    add(pool, 1, (0., 0.), 'no', event_id='one')
    assert add(pool, 1, (0., 0.), 'yes', event_id='one') is None
    add(pool, 2, (.9, 0.), 'yes')
    add(pool, 11, (.9, 0.), 'unknown')
    add(pool, 21, (.9, 0.), 'no')
    add(pool, 22, (1.5, 0.), None)  # close to moving centroid, outside fixed seed radius
    assert len(pool.candidates) == 2
    c = pool.candidates[0]
    assert (c.state, c.yes, c.no, c.sightings, c.n) == ('yes', 1, 2, 2, 4)
    assert c.position == [.9, 0.]
    np.testing.assert_allclose(c.features(), [1., 0., math.log(2), math.log(3), math.log(.5), math.log(3)])
    add(pool, 23, (1.5, 0.), 'unknown')
    assert pool.candidates[1].state == 'unverified'
    for gps in (None, [float('nan'), 0], [1, 2, 3]):
        assert add(pool, 24, gps) is None
    assert len(pool.candidates) == 2


def test_best_yes_view_area_then_recency_and_raw_rgb():
    pool = CandidatePool()
    for step, box, answer in [(1, [1, 1, 5, 5], 'yes'), (2, [1, 1, 9, 9], 'yes'),
                              (3, [1, 1, 9, 9], 'yes'), (4, [0, 0, 19, 19], 'no')]:
        add(pool, step, answer=answer, box=box, rgb=np.full((20, 20, 3), step, dtype=np.uint8))
    candidate = pool.candidates[0]
    assert candidate.best_view['step'] == 3
    assert marked_view(candidate, 'A').getpixel((15, 15)) == (3, 3, 3)


def test_rule_empty_unverified_leader_budget_and_components():
    pool = CandidatePool()
    def decide(t=10, remaining=1000, **kwargs):
        return decision_snapshot(pool, constant_model(), step=t, position=[0, 0], remaining=remaining, **kwargs)
    assert decide()['trigger'] is None
    assert decide(860)['shortlist'] == [] and decide(860)['trigger'] == 'fallback'
    add(pool, gps=(2., 0.), answer='unknown')
    add(pool, gps=(5., 0.))
    assert decide()['leader'] == 0 and decide()['trigger'] is None
    add(pool, 2, gps=(2., 0.), answer='no')
    assert decide()['trigger'] == 'relative_threshold'
    assert decide(remaining=49)['ranked_yes'] == []
    assert decide(remaining=50)['ranked_yes'] == [1]
    assert decide(early_stop=False)['trigger'] is None
    assert decide(drop_rejected=False)['leader'] == 0
    with_u = decision_snapshot(pool, constant_model(c=100), step=10, position=[0, 0], remaining=1000)
    assert with_u['trigger'] is None
    assert decision_snapshot(pool, constant_model(c=100), step=10, position=[0, 0], remaining=1000,
                             use_u=False)['trigger'] == 'relative_threshold'


@pytest.mark.parametrize('n', [1, 2, 3])
def test_pairwise_full_description_call_limit_and_position_bias(n):
    pool, calls = CandidatePool(), []
    for i in range(n):
        add(pool, gps=(i * 2., 0.), rgb=np.zeros((20, 20, 3), dtype=np.uint8), box=[1, 1, 10, 10])
    def complete(prompt, **kwargs):
        calls.append(prompt)
        assert len(kwargs['images']) == 2
        assert 'red upholstery' in prompt and 'beside piano' in prompt
        return '{"choice":"A"}'  # Both directions prefer first image -> tie.
    winner, stats = compare_candidates(SimpleNamespace(complete=complete), pool.candidates, 'chair',
                                       {'intrinsic_attributes': 'red upholstery', 'extrinsic_attributes': 'beside piano'})
    assert winner.candidate_id == 0
    assert len(calls) == stats['logical_requests'] == n * (n - 1)


def test_pairwise_retry_and_request_errors_never_become_uncertainty():
    pool = CandidatePool()
    for i in range(2):
        add(pool, gps=(i * 2., 0.), rgb=np.zeros((20, 20, 3), dtype=np.uint8), box=[1, 1, 10, 10])
    responses = iter(['bad', '{"choice":"B"}', '{"choice":"A"}'])
    winner, stats = compare_candidates(SimpleNamespace(complete=lambda *a, **k: next(responses)),
                                       pool.candidates, 'chair', {})
    assert winner.candidate_id == 1 and stats['format_retries'] == 1
    with pytest.raises(ModelRequestError, match='3 format attempts'):
        compare_candidates(SimpleNamespace(complete=lambda *a, **k: 'bad'), pool.candidates, 'chair', {})
    def fail(*a, **k):
        raise ModelRequestError('offline')
    with pytest.raises(ModelRequestError, match='offline'):
        compare_candidates(SimpleNamespace(complete=fail), pool.candidates, 'chair', {})


def test_prompt_matches_frozen_offline_prompt():
    from tools.pairwise_compare_eval import build_pair_prompt as offline_prompt
    assert build_pair_prompt('chair', 'red', 'window') == offline_prompt('chair', 'red', 'window')


@pytest.fixture
def method(navigation, monkeypatch):
    settings(monkeypatch)
    agent, manager, observations, state = navigation
    agent.args = SimpleNamespace(rag_mode=0)
    agent.count_episodes = 1
    agent.last_gps = np.zeros(2)
    agent.in_stuck_recovery = False
    agent.scenegraph.model_client.backend = 'openai'
    agent.simulator = SimpleNamespace(_env=SimpleNamespace(_elapsed_steps=0, _max_episode_steps=1000))
    agent.get_goal_detection_threshold = lambda: 1
    agent.detection_manager = manager
    observations['gps'] = np.zeros(2)
    manager.reset_instance_verification(SimpleNamespace(scene_id='test-scene'))
    return navigation


def test_navigation_keeps_denied_and_unknown_boxes_but_defers_lock(method):
    agent, manager, obs, state = method
    state['answers'] = ['yes', 'no']
    observe_navigation(method, 1)
    controller = agent.relative_evidence_controller
    assert [c.state for c in controller.pool.candidates] == ['yes', 'no']
    assert not agent.found_goal and len(manager._collect_retired) == 1
    assert state['calls'] == 1
    state['answers'] = ['unknown', 'unknown']
    observe_navigation(method, 2)
    assert [c.state for c in controller.pool.candidates] == ['yes', 'no']
    assert [c.no for c in controller.pool.candidates] == [0, 1]
    manager.reset_post_confirmation_progress(reason='stuck')
    assert [c.yes for c in controller.pool.candidates] == [1, 0]
    manager.reset_instance_verification(SimpleNamespace(scene_id='test-scene'))
    assert not agent.relative_evidence_controller.pool.candidates


@pytest.mark.parametrize('category', ['plant', 'tv_monitor'])
def test_scenegraph_path_joins_method_pool_and_defers_lock(method, category):
    agent, manager, obs, state = method
    agent.obj_goal = category
    agent.obj_goal_sg = 'tv' if category == 'tv_monitor' else category
    state['answers'] = ['yes']
    # The fixture helper emits a caption using obj_goal; align the graph alias for this observation.
    agent.obj_goal_sg = category
    observe_scenegraph(method, 1)
    assert not agent.found_goal
    assert len(manager._collect_retired) == 1
    assert agent.relative_evidence_controller.pool.candidates[0].state == 'yes'


def test_return_target_survives_real_agent_recovery_and_atomic_budget(monkeypatch):
    settings(monkeypatch)
    agent, state, obs = make_agent()
    agent.count_episodes = 1
    agent.goal_attributes = {'intrinsic_attributes': 'red'}
    agent.simulator = SimpleNamespace(_env=SimpleNamespace(_elapsed_steps=974, _max_episode_steps=1000))
    agent.detection_manager = SimpleNamespace(_emit_goal_lock_once=lambda *a, **k: None,
                                             reset_post_confirmation_progress=lambda **k: None,
                                             preserve_evidence_on_recovery_enabled=lambda: False)
    ctrl = RelativeEvidenceNavigation(agent, Calibration(CALIBRATION), 'new-scene')
    agent.relative_evidence_controller = ctrl
    ctrl.model = constant_model()
    add(ctrl.pool, gps=(2., 0.))
    # 26 remaining allows return, although policy step and atomic counts differ.
    assert not ctrl.tick() and ctrl.state == 'returning'
    assert agent.found_goal and ctrl.target == [2., 0.]
    # A locked target must not prevent the existing turn-then-forward recovery sequence.
    agent.forward_no_move_count = agent.not_move_steps = 2
    actions = []
    for angle in (0., 30., 60., 90.):
        agent.full_pose[2] = angle
        actions.append(agent.act(obs)['action'])
    assert actions == [2, 2, 2, 1]
    assert ctrl.state == 'returning' and ctrl.selected == 0
    obs['gps'] = np.array([.1, 0.])
    agent.act(obs)
    assert agent.found_goal and agent.goal_gps.tolist() == [2., 0.]
    assert len(ctrl.pool.candidates) == 1
    state['default_action'] = 0
    obs['gps'] = np.array([.2, 0.])
    assert agent.act(obs)['action'] == 0 and ctrl.state == 'finished'


def test_two_atomic_turns_reduce_return_budget(monkeypatch):
    from utils.atomic_actions import expand_atomic_actions
    settings(monkeypatch)
    agent, _, _ = make_agent()
    agent.count_episodes = 1
    agent.simulator = SimpleNamespace(_env=SimpleNamespace(_elapsed_steps=973, _max_episode_steps=1000))
    ctrl = RelativeEvidenceNavigation(agent, Calibration(CALIBRATION), 'new-scene')
    ctrl.model = constant_model()
    add(ctrl.pool, gps=(2., 0.))
    for _ in expand_atomic_actions({'action': 6}, enabled=True):
        agent.simulator._env._elapsed_steps += 1
    assert not ctrl.tick() and ctrl.state == 'exploring'
    assert ctrl.last_snapshot['remaining_atomic'] == 25
    assert not ctrl.last_snapshot['reachable'][0]


def test_selection_during_recovery_emits_lock_only_after_restoration(method, capsys):
    agent, manager, obs, state = method
    state['answers'] = ['yes', 'no']
    observe_navigation(method, 1)
    ctrl = agent.relative_evidence_controller
    ctrl.model = constant_model()
    agent.in_stuck_recovery = True
    capsys.readouterr()
    assert not ctrl.tick() and ctrl.state == 'returning'
    assert not agent.found_goal
    assert '[Goal Lock]' not in capsys.readouterr().out
    agent.in_stuck_recovery = False
    ctrl.restore_target()
    assert agent.found_goal
    assert agent.goal_gps.tolist() == ctrl.target
    assert '[Goal Lock]' in capsys.readouterr().out
    ctrl.restore_target()
    assert '[Goal Lock]' not in capsys.readouterr().out


def test_disabled_method_does_not_load_calibration(monkeypatch):
    monkeypatch.setenv('ENABLE_RELATIVE_EVIDENCE', 'false')
    monkeypatch.setenv('RELATIVE_EVIDENCE_CALIBRATION', '/missing/calibration')
    assert relative_evidence_run_header() == {'enabled': False}


@pytest.mark.parametrize('key,value', [('VERIFY_BEFORE_CANDIDATE', 'true'),
    ('INSTANCE_COLLECT_ONLY', 'true'), ('INSTANCE_CONFIRMATION_THRESHOLD', '3'),
    ('ENABLE_INSTANCE_CONFIRMATION', 'false'), ('INSTANCE_CONFIRMATION_MODE', 'contradiction'),
    ('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', 'true'),
    ('PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY', 'true'), ('RAG_MODE', '2')])
def test_conflicting_settings_rejected(monkeypatch, key, value):
    settings(monkeypatch)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        validate_relative_evidence(dataset='instancenav', backend='openai')


def test_backend_and_scene_model_selection(monkeypatch):
    settings(monkeypatch)
    with pytest.raises(ValueError, match='multi-image'):
        validate_relative_evidence(backend='openrouter')
    model = Calibration(CALIBRATION)
    scene = next(iter(model.data['scene_folds']))
    assert model.for_scene('/scenes/' + scene)[1].startswith('fold_')
    assert model.for_scene('new-scene')[1] == 'global'


def test_analysis_does_not_mix_new_runs_with_historical_baselines():
    from tools.analyze_verification_set import group_of
    base = {'ENABLE_INSTANCE_CONFIRMATION': 'true', 'VERIFY_BEFORE_CANDIDATE': 'false'}
    assert group_of(base) == 'adm'
    assert group_of({**base, 'ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION': 'false'}) == 'adm_category'
    assert group_of({**base, 'ENABLE_RELATIVE_EVIDENCE': 'true'}) == 'relative'


def test_launcher_dual_workers_receive_cli_overrides_and_log_them(tmp_path):
    script = tmp_path / 'scripts/run_instance100.sh'
    script.parent.mkdir()
    script.write_text((ROOT / 'scripts/run_instance100.sh').read_text())
    fake = tmp_path / 'python'
    fake.write_text(f'#!{sys.executable}\n' + '''
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
if args == ['-']:
    source = sys.stdin.read()
    if 'relative_evidence_run_header' in source:
        print(json.dumps({'enabled': True, 'calibration_sha256': 'test'}))
elif '-c' in args:
    print(time.time())
elif 'PSG_Nav.py' in args:
    gpu = args[args.index('--worker-gpu') + 1]
    Path('worker_' + gpu + '.json').write_text(json.dumps({
        k: v for k, v in os.environ.items() if k.startswith('RELATIVE_EVIDENCE_') or k == 'ENABLE_RELATIVE_EVIDENCE'}))
    time.sleep(.2)
''')
    fake.chmod(0o755)
    env = {**os.environ, 'PSGNAV_PYTHON': str(fake), 'ENABLE_RELATIVE_EVIDENCE': 'false',
           'ENABLE_INSTANCE_CONFIRMATION': 'true', 'VERIFY_BEFORE_CANDIDATE': 'false',
           'INSTANCE_CONFIRMATION_THRESHOLD': '1', 'INSTANCE_CONFIRMATION_MODE': 'positive_match',
           'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK': 'false',
           'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY': 'false', 'INSTANCE_COLLECT_ONLY': 'false', 'RAG_MODE': '0'}
    env.pop('ENABLE_CONTRADICTION_CHECK', None)
    result = subprocess.run(['bash', str(script), '--enable-relative-evidence', 'YES',
                             '--relative-evidence-use-u', 'NO', '--relative-evidence-early-stop', 'false',
                             '--relative-evidence-pairwise', '0', '--relative-evidence-drop-rejected', 'no',
                             '--relative-evidence-calibration', 'model with spaces.json', '--timestamp', 'test'],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    for gpu in (0, 1):
        captured = json.loads((tmp_path / f'worker_{gpu}.json').read_text())
        assert captured['ENABLE_RELATIVE_EVIDENCE'] == 'true'
        assert captured['RELATIVE_EVIDENCE_CALIBRATION'] == 'model with spaces.json'
        assert all(captured['RELATIVE_EVIDENCE_' + key] == 'false'
                   for key in ('USE_U', 'EARLY_STOP', 'PAIRWISE', 'DROP_REJECTED'))
        log = (tmp_path / f'outputs/instancenav_val10_100_test_gpu{gpu}.log').read_text()
        assert 'ENABLE_RELATIVE_EVIDENCE: true' in log and 'calibration_sha256' in log
    for options in (['--enable-relative-evidence', 'bad'], ['--relative-evidence-calibration'],
                    ['--relative-evidence-use-u'], ['--enable-relative-evidence', 'true', '--verify-before-candidate', 'true']):
        result = subprocess.run(['bash', str(script), *options], env=env, capture_output=True, text=True, timeout=5)
        assert result.returncode == 2
