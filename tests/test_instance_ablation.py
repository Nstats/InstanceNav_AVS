import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from configs.navigation_config import validate_instance_ablation
from tools.run_instance_ablation import settings, launch_command, archive_run, source_inventory
from test_instance_constraints import navigation, post_navigation, episode, Predictions, observe_navigation
from test_instance_confirmation_controls import controls, observe_scenegraph
from utils.instance_verifier import verify_candidates
from utils.model_client import ModelRequestError


@pytest.fixture
def ablation(controls, monkeypatch):
    monkeypatch.setenv('VERIFY_BEFORE_CANDIDATE', 'false')
    monkeypatch.setenv('INSTANCE_VERIFICATION_SCOPE', 'instance')
    monkeypatch.setenv('INSTANCE_REJECTION_POLICY', 'none')
    monkeypatch.setenv('INSTANCE_VERIFICATION_PROMPT', 'match_strict_no')
    monkeypatch.setenv('INSTANCE_VERIFICATION_RECORD_INPUTS', 'false')
    monkeypatch.setenv('INSTANCE_CONFIRMATION_COUNT_UNIT', 'frame')
    return controls


@pytest.mark.parametrize('scope', ['instance', 'category'])
def test_category_prompt_has_no_instance_description_and_keeps_ternary_contract(monkeypatch, scope):
    monkeypatch.setenv('INSTANCE_VERIFICATION_SCOPE', scope)
    calls = []
    def complete(prompt, image, **kwargs):
        calls.append(prompt)
        answer = '{"matches":["no","unknown","yes"]}'
        kwargs['validator'](answer)
        return answer
    kept = verify_candidates(SimpleNamespace(complete=complete), np.zeros((30, 30, 3), dtype=np.uint8),
                             [[1, 1, 5, 5], [6, 6, 10, 10], [12, 12, 20, 20]], 'chair',
                             {'intrinsic_attributes': 'SECRET_RED_STRIPES', 'extrinsic_attributes': 'SECRET_NEXT_TO_DESK',
                              'goal_object_id': 'ORACLE_ID', 'image': 'ORACLE_IMAGE'},
                             log_context={'stage': 'goal_confirmation'})
    assert kept == [2] and len(calls) == 1
    assert calls[0].startswith('Stage: goal confirmation.')
    assert 'ORACLE_' not in calls[0]
    assert ('SECRET_' in calls[0]) == (scope == 'instance')
    assert ('Matching the object category alone is not sufficient.' in calls[0]) == (scope == 'instance')


@pytest.mark.parametrize('path', ['navigation', 'scenegraph'])
@pytest.mark.parametrize('policy,answer,veto', [
    ('none', 'no', False), ('none', 'unknown', False),
    ('no', 'no', True), ('no', 'unknown', False),
    ('non_yes', 'no', True), ('non_yes', 'unknown', True),
])
def test_ternary_veto_survives_recovery_but_not_episode_reset(ablation, monkeypatch, path, policy, answer, veto):
    agent, manager, _, state = ablation
    monkeypatch.setenv('INSTANCE_REJECTION_POLICY', policy)
    observe = observe_navigation if path == 'navigation' else observe_scenegraph
    state['stage_answers']['goal_confirmation'] = [answer]
    observe(ablation, 1)
    assert not agent.found_goal
    assert bool(agent.instance_rejections) == veto
    if veto:
        assert agent.instance_rejections[0]['decision'] == answer
    manager.reset_post_confirmation_progress(reason='stuck')
    state['stage_answers']['goal_confirmation'] = ['yes']
    observe(ablation, 2)
    assert agent.found_goal == (not veto)
    assert len(state['stage_calls']) == (1 if veto else 2)
    manager.reset_instance_verification(episode())
    agent.found_goal = False
    observe(ablation, 3)
    assert agent.found_goal  # Same object is allowed in a new episode.


@pytest.mark.parametrize('path', ['navigation', 'scenegraph'])
def test_no_veto_clears_existing_votes_without_adding_negative_track(ablation, monkeypatch, path):
    agent, manager, _, state = ablation
    monkeypatch.setenv('INSTANCE_CONFIRMATION_THRESHOLD', '2')
    monkeypatch.setenv('INSTANCE_REJECTION_POLICY', 'no')
    observe = observe_navigation if path == 'navigation' else observe_scenegraph
    observe(ablation, 1)
    track = manager._positive_evidence.tracks[0]
    assert len(track.votes) == 1 and not agent.found_goal
    state['stage_answers']['goal_confirmation'] = ['no']
    observe(ablation, 2)
    assert track.vetoed and not track.votes and not agent.found_possible_goal
    assert len(manager._positive_evidence.tracks) == 1


@pytest.mark.parametrize('path', ['navigation', 'scenegraph'])
def test_unknown_does_not_reset_positive_votes_for_e5(ablation, monkeypatch, path):
    agent, manager, _, state = ablation
    monkeypatch.setenv('INSTANCE_CONFIRMATION_THRESHOLD', '2')
    observe = observe_navigation if path == 'navigation' else observe_scenegraph
    observe(ablation, 1)
    state['stage_answers']['goal_confirmation'] = ['unknown']
    observe(ablation, 2)
    assert not agent.found_goal and len(manager._positive_evidence.tracks[0].votes) == 1
    state['stage_answers']['goal_confirmation'] = ['yes']
    observe(ablation, 3)
    assert agent.found_goal and len(manager._positive_evidence.tracks[0].votes) == 2


@pytest.mark.parametrize('path', ['navigation', 'scenegraph'])
def test_reprocessing_same_frame_does_not_supply_second_vote(ablation, monkeypatch, path):
    agent, manager, _, _ = ablation
    monkeypatch.setenv('INSTANCE_CONFIRMATION_THRESHOLD', '2')
    observe = observe_navigation if path == 'navigation' else observe_scenegraph
    observe(ablation, 1)
    observe(ablation, 1)
    assert len(manager._positive_evidence.tracks[0].votes) == 1 and not agent.found_goal
    observe(ablation, 2)
    assert agent.found_goal


def test_no_in_duplicate_box_vetoes_yes_for_same_position(ablation, monkeypatch):
    agent, manager, _, state = ablation
    monkeypatch.setenv('INSTANCE_REJECTION_POLICY', 'no')
    agent.current_obj_predictions = Predictions([[270, 220, 330, 280]] * 2)
    state['stage_answers']['goal_confirmation'] = ['yes', 'no']
    observe_navigation(ablation, 1)
    assert not agent.found_goal and not manager._positive_evidence.tracks
    assert len(agent.instance_rejections) == 1


def test_no_rule_preserves_other_candidate_and_original_box_indices(ablation, monkeypatch):
    agent, manager, _, state = ablation
    monkeypatch.setenv('INSTANCE_REJECTION_POLICY', 'no')
    agent.current_obj_predictions = Predictions([[270, 220, 330, 280], [470, 220, 530, 280]])
    state['stage_answers']['goal_confirmation'] = ['no', 'unknown']
    observe_navigation(ablation, 1)
    assert len(agent.instance_rejections) == 1
    state['stage_answers']['goal_confirmation'] = ['yes']
    observe_navigation(ablation, 2)
    assert agent.found_goal and manager._positive_evidence.tracks[0].box == [470, 220, 530, 280]
    assert len(state['stage_calls']) == 2


def test_malformed_batch_never_commits_partial_no(ablation, monkeypatch):
    agent, manager, _, state = ablation
    monkeypatch.setenv('INSTANCE_REJECTION_POLICY', 'non_yes')
    state['stage_answers']['goal_confirmation'] = ['no', 'invalid']
    with pytest.raises(ModelRequestError):
        observe_navigation(ablation, 1)
    assert not agent.instance_rejections and not manager._positive_evidence.tracks


@pytest.mark.parametrize('override', [
    {'INSTANCE_VERIFICATION_SCOPE': 'typo'}, {'INSTANCE_REJECTION_POLICY': 'typo'},
    {'INSTANCE_VERIFICATION_SCOPE': 'category', 'INSTANCE_CONFIRMATION_MODE': 'contradiction'},
    {'INSTANCE_REJECTION_POLICY': 'no', 'VERIFY_BEFORE_CANDIDATE': 'true'},
    {'INSTANCE_REJECTION_POLICY': 'no', 'ENABLE_INSTANCE_CONFIRMATION': 'false'},
])
def test_invalid_combinations_fail_before_navigation(monkeypatch, override):
    for key, value in {**settings('B'), **override}.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        validate_instance_ablation()


def test_request_recording_saves_actual_image_prompt_and_raw_answer(tmp_path, monkeypatch):
    monkeypatch.setenv('INSTANCE_VERIFICATION_RECORD_INPUTS', 'true')
    monkeypatch.setenv('PSGNAV_LOG_FILE', str(tmp_path / 'worker.log'))
    client = SimpleNamespace(complete=lambda *a, **k: '{"matches":["unknown"]}')
    verify_candidates(client, np.zeros((20, 20, 3), dtype=np.uint8), [[1, 1, 5, 5]], 'chair',
                      {'intrinsic_attributes': 'red chair'})
    directory = tmp_path / 'worker.verification'
    records = [json.loads(l) for l in (directory / 'events.jsonl').read_text().splitlines()]
    request = next(r for r in records if r['event'] == 'request')
    assert (directory / request['image']['path']).exists()
    assert 'red chair' in request['prompt']
    assert next(r for r in records if r['event'] == 'result')['matches'] == ['unknown']


def test_fixed_presets_differ_only_by_planned_variable():
    base = settings('B')
    expected = {'E1': {'ENABLE_INSTANCE_CONFIRMATION': 'false'},
                'E2': {'INSTANCE_VERIFICATION_SCOPE': 'category'},
                'E3': {'INSTANCE_REJECTION_POLICY': 'no'},
                'E4': {'INSTANCE_REJECTION_POLICY': 'non_yes'},
                'E5': {'INSTANCE_CONFIRMATION_THRESHOLD': '2'}}
    for experiment, diff in expected.items():
        assert {k: v for k, v in settings(experiment).items() if v != base[k]} == diff
        cmd = launch_command(experiment, '2,3', 'trial', True)
        assert '--resume' in cmd and '--gpus' in cmd


def test_manifest_blocks_mixed_sources_and_mismatched_resume(tmp_path, monkeypatch):
    import tools.run_instance_ablation as launcher
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    (tmp_path / 'source.py').write_text('x = 1\n')
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *a, **k: b'test-head')
    inventory = (['source.py'], {'source.py': 'digest'}, '', 'suite-digest')
    args = SimpleNamespace(experiment='E3', gpus='0,1', timestamp='trial', suite='suite', resume=False)
    directory = tmp_path / 'outputs/instance_ablation_runs/suite/trial'
    path = archive_run(args, ['bash'], directory, inventory)
    assert path.exists() and (directory / 'source.tar.gz').exists()
    args.resume = True
    assert archive_run(args, ['bash'], directory, inventory) == path
    args.experiment = 'E4'
    with pytest.raises(ValueError, match='Resume'):
        archive_run(args, ['bash'], directory, inventory)
    args.resume, args.timestamp = False, 'second'
    with pytest.raises(ValueError, match='Suite source'):
        archive_run(args, ['bash'], directory.parent / 'second', (*inventory[:3], 'changed'))
