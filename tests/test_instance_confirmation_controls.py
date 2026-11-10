import json
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from agent_test_support import load_agent_methods
from configs.navigation_config import (
    instance_confirmation_enabled, instance_confirmation_threshold,
    verify_before_candidate_enabled, post_confirmation_contradiction_check_enabled,
    preserve_positive_evidence_on_recovery_enabled,
    group_llm_filter_enabled, room_llm_filter_enabled,
)
from test_instance_constraints import navigation, post_navigation, Predictions, episode, observe_navigation
from utils.model_client import ModelClient, ModelRequestError


@pytest.fixture
def controls(post_navigation, monkeypatch):
    agent, manager, observations, state = post_navigation
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'true')
    monkeypatch.setenv('INSTANCE_CONFIRMATION_THRESHOLD', '1')
    monkeypatch.setenv('VERIFY_BEFORE_CANDIDATE', 'true')
    monkeypatch.setenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', 'false')
    manager.reset_instance_verification(episode())
    state.update(stage_calls=[], stage_answers={'candidate_admission': ['yes'], 'goal_confirmation': ['yes']})

    def complete(prompt, image, **options):
        stage = 'candidate_admission' if prompt.startswith('Stage: candidate admission.') else 'goal_confirmation'
        state['stage_calls'].append((stage, prompt))
        if state.get('fail_stage') == stage:
            raise ModelRequestError('Independent check failed')
        response = json.dumps({'matches': state['stage_answers'][stage]})
        options['validator'](response)
        return response

    state['primary_complete'] = complete
    return post_navigation


def observe_scenegraph(fixture, step):
    agent, manager, observations, _ = fixture
    agent.total_steps = step
    observations['rgb'].fill(step)
    if not hasattr(agent.scenegraph, 'nodes'):
        agent.scenegraph.nodes = [SimpleNamespace(is_goal_node=True, object={'id': 7, 'image_idx': [], 'mask_idx': []})]
        agent.scenegraph.segment2d_results = []
    node = agent.scenegraph.nodes[0]
    node.object['image_idx'].append(len(agent.scenegraph.segment2d_results))
    node.object['mask_idx'].append(0)
    mask = np.zeros((480, 640), dtype=bool)
    mask[220:280, 270:330] = True
    agent.scenegraph.segment2d_results.append({'caption': [agent.obj_goal], 'mask': [mask]})
    manager._detect_goal_with_scenegraph(observations)


def test_new_defaults_and_legacy_master_alias(monkeypatch):
    for name in ('ENABLE_INSTANCE_CONFIRMATION', 'ENABLE_INSTANCE_CONFIRMATION_BEFORE_FIND_GOAL',
                 'INSTANCE_CONFIRMATION_THRESHOLD', 'VERIFY_BEFORE_CANDIDATE'):
        monkeypatch.delenv(name, raising=False)
    assert instance_confirmation_enabled() and verify_before_candidate_enabled()
    assert instance_confirmation_threshold() == 1
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION_BEFORE_FIND_GOAL', 'false')
    with pytest.warns(DeprecationWarning, match='deprecated'):
        assert not instance_confirmation_enabled()
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'true')
    assert instance_confirmation_enabled()  # Explicit canonical name takes precedence.


@pytest.mark.parametrize('value', ['0', '-1', '1.5', 'bad'])
def test_threshold_requires_positive_integer(monkeypatch, value):
    monkeypatch.setenv('INSTANCE_CONFIRMATION_THRESHOLD', value)
    with pytest.raises(ValueError, match='positive integer'):
        instance_confirmation_threshold()


@pytest.mark.parametrize('enabled', [False, True])
def test_group_and_room_filters_are_independent_of_instance_master(monkeypatch, enabled):
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', str(enabled).lower())
    monkeypatch.setenv('ENABLE_GROUP_LLM_FILTER', 'true')
    monkeypatch.setenv('ENABLE_ROOM_LLM_FILTER', 'false')
    assert group_llm_filter_enabled() and not room_llm_filter_enabled()
    monkeypatch.setenv('ENABLE_GROUP_LLM_FILTER', 'false')
    monkeypatch.setenv('ENABLE_ROOM_LLM_FILTER', 'true')
    assert not group_llm_filter_enabled() and room_llm_filter_enabled()


@pytest.mark.parametrize('path', ['navigation', 'scenegraph'])
def test_admission_pass_is_not_a_confirmation_vote(controls, path):
    agent, manager, _, state = controls
    state['stage_answers']['goal_confirmation'] = ['no']
    (observe_navigation if path == 'navigation' else observe_scenegraph)(controls, 1)
    assert [stage for stage, _ in state['stage_calls']] == ['candidate_admission', 'goal_confirmation']
    assert not agent.found_goal and not agent.goal_gps_map.any()
    assert not any(track.votes for track in manager._positive_evidence.tracks)


def test_admission_denial_never_starts_confirmation_counting(controls):
    agent, manager, _, state = controls
    state['stage_answers']['candidate_admission'] = ['no']
    observe_navigation(controls, 1)
    assert [stage for stage, _ in state['stage_calls']] == ['candidate_admission']
    assert not agent.found_goal and not agent.found_possible_goal
    assert not manager._positive_evidence.tracks


def test_confirmation_indices_still_refer_to_original_boxes(controls):
    agent, manager, _, state = controls
    agent.current_obj_predictions = Predictions([[270, 220, 330, 280], [400, 220, 460, 280]])
    state['stage_answers'] = {'candidate_admission': ['no', 'yes'], 'goal_confirmation': ['yes']}
    observe_navigation(controls, 1)
    assert agent.found_goal
    assert len(manager._positive_evidence.tracks) == 1
    track = manager._positive_evidence.tracks[0]
    assert track.box == [400, 220, 460, 280] and len(track.votes) == 1


def test_node_veto_clears_candidate_votes_and_approach_even_without_current_depth(controls, monkeypatch):
    agent, manager, _, state = controls
    monkeypatch.setenv('INSTANCE_CONFIRMATION_MODE', 'contradiction')
    monkeypatch.setenv('INSTANCE_CONFIRMATION_THRESHOLD', '3')
    state['stage_answers'] = {'candidate_admission': ['no'], 'goal_confirmation': ['no']}
    observe_scenegraph(controls, 1)
    track = manager._positive_evidence.tracks[0]
    assert agent.found_possible_goal and len(track.votes) == 1
    agent.depth.fill(100)
    state['stage_answers']['candidate_admission'] = ['yes']
    observe_scenegraph(controls, 2)
    assert track.vetoed and not track.votes
    assert not agent.goal_gps_map.any() and not agent.found_possible_goal
    assert manager._post_possible_track_id is None


def test_exact_api_cache_cannot_reuse_admission_as_confirmation(controls, monkeypatch):
    agent, manager, _, _ = controls
    monkeypatch.setenv('PSGNAV_MODEL_BACKEND', 'openai')
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-key')
    client = ModelClient()
    payloads = []

    def send(payload):
        payloads.append(payload)
        return json.dumps({'matches': ['yes' if len(payloads) == 1 else 'no']})

    monkeypatch.setattr(client, '_send', send)
    agent.scenegraph.model_client = client
    observe_navigation(controls, 1)
    assert len(payloads) == 2 and payloads[0] != payloads[1]
    assert not agent.found_goal and not agent.goal_gps_map.any()


@pytest.mark.parametrize('category,path', [('chair', 'navigation'), ('bed', 'navigation'),
                                        ('sofa', 'navigation'), ('toilet', 'navigation'),
                                        ('plant', 'scenegraph'), ('tv_monitor', 'scenegraph')])
@pytest.mark.parametrize('threshold', [1, 3])
def test_all_categories_lock_at_shared_threshold(controls, monkeypatch, category, path, threshold):
    agent, manager, _, state = controls
    monkeypatch.setenv('INSTANCE_CONFIRMATION_THRESHOLD', str(threshold))
    agent.obj_goal = agent.obj_goal_sg = category
    agent.current_obj_predictions.fields['labels'] = [category]
    observe = observe_navigation if path == 'navigation' else observe_scenegraph
    for step in range(1, threshold + 1):
        observe(controls, step)
        assert len(manager._positive_evidence.tracks[0].votes) == step
        assert agent.found_goal == (step == threshold)
        assert agent.found_possible_goal == (step < threshold)
    assert len(state['stage_calls']) == 2 * threshold
    assert agent.found_goal_times == threshold


@pytest.mark.parametrize('mode,answer,accepted', [
    ('positive_match', 'yes', True), ('positive_match', 'no', False),
    ('positive_match', 'unknown', False), ('contradiction', 'no', True),
    ('contradiction', 'unknown', True), ('contradiction', 'yes', False),
])
@pytest.mark.parametrize('path', ['navigation', 'scenegraph'])
def test_mode_controls_confirmation_vote_meaning(controls, monkeypatch, mode, answer, accepted, path):
    agent, manager, _, state = controls
    monkeypatch.setenv('INSTANCE_CONFIRMATION_MODE', mode)
    monkeypatch.setenv('INSTANCE_CONFIRMATION_THRESHOLD', '2')
    state['stage_answers']['candidate_admission'] = ['no' if mode == 'contradiction' else 'yes']
    state['stage_answers']['goal_confirmation'] = [answer]
    observe = observe_navigation if path == 'navigation' else observe_scenegraph
    observe(controls, 1)
    assert not agent.found_goal
    observe(controls, 2)
    assert agent.found_goal == accepted
    assert sum(len(t.votes) for t in manager._positive_evidence.tracks) == (2 if accepted else 0)


@pytest.mark.parametrize('preverify', [False, True])
def test_far_candidate_admission_never_counts_toward_lock(controls, monkeypatch, preverify):
    agent, manager, _, state = controls
    monkeypatch.setenv('VERIFY_BEFORE_CANDIDATE', str(preverify).lower())
    agent.depth.fill(4.2)
    observe_navigation(controls, 1)
    assert agent.found_possible_goal and not agent.found_goal
    assert not manager._positive_evidence.tracks[0].votes
    assert len(state['stage_calls']) == int(preverify)
    agent.depth.fill(3.8)
    observe_navigation(controls, 2)
    assert agent.found_goal and len(manager._positive_evidence.tracks[0].votes) == 1
    assert len(state['stage_calls']) == (3 if preverify else 1)


@pytest.mark.parametrize('mode', ['positive_match', 'contradiction'])
@pytest.mark.parametrize('path', ['navigation', 'scenegraph'])
def test_master_off_bypasses_all_instance_checks_and_new_threshold(controls, monkeypatch, mode, path):
    agent, manager, _, _ = controls
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'false')
    monkeypatch.setenv('INSTANCE_CONFIRMATION_MODE', mode)
    monkeypatch.setenv('INSTANCE_CONFIRMATION_THRESHOLD', '99')
    monkeypatch.setenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', 'true')
    monkeypatch.setenv('PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY', 'true')
    manager.reset_instance_verification(episode())
    fail = lambda *args, **kwargs: pytest.fail('Master off must bypass every instance verification')
    agent.scenegraph.model_client.complete = fail
    agent.rag_check_enabled = True
    agent.rag_manager = SimpleNamespace(verify_detection=fail)
    assert not verify_before_candidate_enabled()
    assert not post_confirmation_contradiction_check_enabled()
    assert not preserve_positive_evidence_on_recovery_enabled()
    # The historical detector-only paths remain five for this chair fixture,
    # and direct near locking for scenegraph objects.
    steps = 5 if path == 'navigation' else 1
    observe = observe_navigation if path == 'navigation' else observe_scenegraph
    for step in range(1, steps + 1):
        observe(controls, step)
        assert agent.found_goal == (step == steps)
    assert manager._positive_evidence is None


def test_real_agent_threshold_leaves_scenegraph_filter_and_disabled_baseline_unchanged(monkeypatch):
    agent_class = load_agent_methods('get_goal_detection_threshold')
    agent = SimpleNamespace(dataset='instancenav', obj_goal='chair', obj_goal_sg='chair',
                            scenegraph=SimpleNamespace(threshold_list={'chair': 5},
                                                       cfg=SimpleNamespace(obj_min_detections=5)))
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'true')
    monkeypatch.setenv('INSTANCE_CONFIRMATION_THRESHOLD', '2')
    assert agent_class.get_goal_detection_threshold(agent) == 2
    assert agent.scenegraph.threshold_list == {'chair': 5} and agent.scenegraph.cfg.obj_min_detections == 5
    agent.dataset = 'mp3d'
    assert agent_class.get_goal_detection_threshold(agent) == 5
    agent.dataset = 'instancenav'
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'false')
    assert agent_class.get_goal_detection_threshold(agent) == 5


def test_other_navigation_datasets_keep_rag_verification(controls, monkeypatch):
    agent, manager, _, _ = controls
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'false')
    agent.dataset = 'mp3d'
    calls = []
    def verify(**kwargs):
        calls.append(kwargs)
        return True, .9, 'Verified', None
    agent.rag_manager = SimpleNamespace(verify_detection=verify)
    agent.rag_latest_goal_detection = {'crop': None, 'caption': 'chair', 'confidence': .8}
    manager._verify_detection_with_rag()
    assert len(calls) == 1 and agent.rag_latest_goal_detection['verified']


@pytest.mark.parametrize('preverify', [False, True])
def test_neighbors_cannot_pool_votes_without_recovery_preservation(controls, monkeypatch, preverify):
    agent, manager, _, state = controls
    monkeypatch.setenv('INSTANCE_CONFIRMATION_THRESHOLD', '3')
    monkeypatch.setenv('VERIFY_BEFORE_CANDIDATE', str(preverify).lower())
    agent.current_obj_predictions = Predictions([[270, 220, 330, 280], [335, 220, 395, 280]])
    state['stage_answers'] = {'candidate_admission': ['yes', 'yes'], 'goal_confirmation': ['yes', 'yes']}
    for step in (1, 2):
        observe_navigation(controls, step)
    assert agent.goal_gps_map.max() == 4 and not agent.found_goal
    assert [len(t.votes) for t in manager._positive_evidence.tracks] == [2, 2]
    observe_navigation(controls, 3)
    assert agent.found_goal and agent.found_goal_times == 3


@pytest.mark.parametrize('stage', ['candidate_admission', 'goal_confirmation'])
def test_failed_stage_never_becomes_a_vote(controls, stage):
    agent, manager, _, state = controls
    state['fail_stage'] = stage
    with pytest.raises(ModelRequestError, match='Independent check failed'):
        observe_navigation(controls, 1)
    assert not agent.found_goal and not agent.goal_gps_map.any()
    assert not manager._positive_evidence.tracks


def test_launcher_exports_controls_and_accepts_legacy_cli_alias(tmp_path):
    fake = tmp_path / 'python'
    fake.write_text('#!/bin/bash\nif [[ "$1" == "-" ]]; then cat >/dev/null; fi\n'
                    'printf "%s %s %s\\n" "$ENABLE_INSTANCE_CONFIRMATION" '
                    '"$INSTANCE_CONFIRMATION_THRESHOLD" "$VERIFY_BEFORE_CANDIDATE" >> "$CONTROL_CAPTURE"\n')
    fake.chmod(0o755)
    capture = tmp_path / 'capture'
    env = {**os.environ, 'PSGNAV_PYTHON': str(fake), 'CONTROL_CAPTURE': str(capture),
           'ENABLE_INSTANCE_CONFIRMATION_BEFORE_FIND_GOAL': 'true',
           'INSTANCE_CONFIRMATION_MODE': 'positive_match',
           'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK': 'false',
           'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY': 'false'}
    env.pop('ENABLE_CONTRADICTION_CHECK', None)
    env.pop('ENABLE_INSTANCE_CONFIRMATION', None)
    command = ['bash', 'scripts/run_instance100.sh', '--check-only',
               '--enable-instance-confirmation-before-find-goal', 'no',
               '--instance-confirmation-threshold', '3', '--verify-before-candidate', 'NO']
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == ['false 3 false'] * 2
    assert 'INSTANCE_CONFIRMATION_THRESHOLD: 3' in result.stdout
    assert 'VERIFY_BEFORE_CANDIDATE: false' in result.stdout
    for options in (['--instance-confirmation-threshold', '0'], ['--instance-confirmation-threshold'],
                    ['--verify-before-candidate', 'bad'], ['--verify-before-candidate']):
        result = subprocess.run(['bash', 'scripts/run_instance100.sh', *options], env=env,
                                capture_output=True, text=True)
        assert result.returncode == 2, options
