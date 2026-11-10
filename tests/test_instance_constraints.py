import copy
import csv
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent_test_support import load_agent_methods
from configs.navigation_config import (instance_confirmation_mode, post_confirmation_contradiction_check_enabled,
                                       preserve_positive_evidence_on_recovery_enabled)
from utils.instance_verifier import (
    DEFAULT_FACTOR_FILE, FACTOR_STATUSES, build_constraint_prompt,
    constraints_for_episode, parse_constraint_results, verify_candidate_constraints,
)
from utils.model_client import ModelRequestError
from utils.object_detection import ObjectDetectionManager


def verification_records(capsys):
    prefix = '[Instance Verification] '
    return [json.loads(line[len(prefix):]) for line in capsys.readouterr().out.splitlines()
            if line.startswith(prefix)]


def episode():
    return SimpleNamespace(goals_key='mv2HUxq3B53_72', episode_id='635', object_category='chair')  # gitleaks:allow (non-secret package version or test fixture ID)


def result_for(factors, statuses):
    return {'candidates': [
        {'candidate_id': i, 'constraints': [
            {'factor_id': factor['factor_id'], 'status': status, 'evidence': 'Visible test evidence.'}
            for factor, status in zip(factors, values)]}
        for i, values in enumerate(statuses)]}


def test_all_csv_tasks_use_stable_keys_and_cached_rows_are_not_mutated():
    with DEFAULT_FACTOR_FILE.open() as stream:
        rows = list(csv.DictReader(stream))
    tasks = {}
    for row in rows:
        tasks.setdefault(row['episode_key'], []).append(row)
    assert len(tasks) == 100
    for key, expected in reversed(list(tasks.items())):
        goals_key, episode_id = key.rsplit('__ep', 1)
        ep = SimpleNamespace(goals_key=goals_key, episode_id=episode_id,
                             object_category=expected[0]['category'])
        factors = constraints_for_episode(ep)
        assert [f['factor_id'] for f in factors] == [f['factor_id'] for f in expected]
        assert [f['source_span'] for f in factors] == [f['source_span'] for f in expected]
        factors[0]['value'] = 'mutated'
        assert constraints_for_episode(ep)[0]['value'] == expected[0]['value']
    with pytest.raises(ValueError, match='No subconstraints'):
        constraints_for_episode(SimpleNamespace(goals_key='missing', episode_id='1', object_category='chair'))
    with pytest.raises(ValueError, match='category mismatch'):
        constraints_for_episode(SimpleNamespace(goals_key=episode().goals_key, episode_id='635', object_category='sofa'))


@pytest.mark.parametrize('factor_count,candidate_count', [(1, 1), (2, 1), (5, 2)])
def test_english_prompt_examples_are_complete_and_never_contain_oracle_data(factor_count, candidate_count):
    factors = constraints_for_episode(episode())[:factor_count]
    attributes = {'intrinsic_attributes': 'A beige chair.', 'extrinsic_attributes': 'Next to a table.',
                  'image': 'PRIVATE_TARGET_IMAGE', 'goal_object_id': 'PRIVATE_OBJECT_ID'}
    prompt = build_constraint_prompt('chair', attributes, factors, candidate_count)
    assert prompt.isascii()
    assert 'A beige chair.' in prompt and 'Next to a table.' in prompt
    assert 'PRIVATE_' not in prompt and episode().goals_key not in prompt
    assert 'brownish-yellow' in prompt or factor_count < 3
    examples = prompt.split('FORMAT EXAMPLES (fictional, alternative responses; structure only):\n')[1]
    states = set()
    while examples.lstrip().startswith('{'):
        examples = examples.lstrip()
        data, end = json.JSONDecoder().raw_decode(examples)
        parsed = parse_constraint_results(json.dumps(data), candidate_count, [f['factor_id'] for f in factors])
        states.update(f['status'] for candidate in parsed for f in candidate['constraints'])
        examples = examples[end:]
    assert states == set(FACTOR_STATUSES)
    assert 'not evidence about the supplied image' in examples


@pytest.mark.parametrize('mutation', [
    lambda r: r['candidates'].pop(),
    lambda r: r['candidates'][0].update(candidate_id=True),
    lambda r: r['candidates'][1].update(candidate_id=0),
    lambda r: r['candidates'][0]['constraints'].pop(),
    lambda r: r['candidates'][0]['constraints'][1].update(factor_id='a1'),
    lambda r: r['candidates'][0]['constraints'][0].update(status='yes'),
    lambda r: r['candidates'][0]['constraints'][0].update(evidence=' '),
    lambda r: r['candidates'][0].update(overall='yes'),
])
def test_malformed_factor_results_fail_without_filling_in_missing_evidence(mutation):
    factors = constraints_for_episode(episode())[:2]
    response = result_for(factors, [['unknown'] * 2, ['not_contradicted'] * 2])
    mutation(response)
    with pytest.raises(ValueError):
        parse_constraint_results(json.dumps(response), 2, ['a1', 'a2'])


def test_duplicate_json_status_cannot_hide_a_contradiction():
    response = ('{"candidates":[{"candidate_id":0,"constraints":[{"factor_id":"a1",'
                '"status":"contradicted","status":"unknown","evidence":"Blue cushion."}]}]}')
    with pytest.raises(ValueError, match='Duplicate JSON field'):
        parse_constraint_results(response, 1, ['a1'])


def test_factor_verification_batches_candidates_and_retries_only_bad_format(capsys):
    factors = constraints_for_episode(episode())
    calls = []
    response = result_for(factors, [['unknown'] * 5, ['not_contradicted'] * 4 + ['contradicted']])

    def complete(prompt, image, **options):
        calls.append((prompt, image.tobytes()))
        schema = options['response_format']['json_schema']['schema']
        assert schema['properties']['candidates']['minItems'] == 2
        answer = '{"candidates":[]}' if len(calls) == 1 else json.dumps(response)
        options['validator'](answer)
        return answer

    actual = verify_candidate_constraints(SimpleNamespace(complete=complete), np.zeros((20, 20, 3), dtype=np.uint8),
                                         [[1, 1, 5, 5], [8, 8, 15, 15]], 'chair',
                                         {'intrinsic_attributes': 'A beige chair.'}, factors,
                                         log_context={'step': 42, 'box_indices': [1, 3]})
    assert actual == response['candidates']
    assert len(calls) == 2 and calls[0][0] != calls[1][0] and calls[0][1] == calls[1][1]
    records = verification_records(capsys)
    assert [r['event'] for r in records] == ['raw_response', 'format_error', 'raw_response', 'result']
    assert [r['attempt'] for r in records] == [1, 1, 2, 2]
    assert all(r['verification_id'] for r in records)
    assert all(r['mode'] == 'subconstraint' and r['step'] == 42 and r['box_indices'] == [1, 3]
               for r in records)
    assert records[0]['answer'] == '{"candidates":[]}'
    assert records[2]['answer'] == json.dumps(response)
    assert records[3]['candidates'] == response['candidates']


class Predictions:
    def __init__(self, boxes):
        self.bbox = torch.tensor(boxes)
        self.fields = {'labels': ['chair'] * len(boxes), 'scores': torch.ones(len(boxes)) * .9}

    def get_field(self, field):
        return self.fields[field]


@pytest.fixture
def navigation(monkeypatch):
    monkeypatch.delenv('ENABLE_INSTANCE_CONFIRMATION_BEFORE_FIND_GOAL', raising=False)
    monkeypatch.setenv('VERIFY_BEFORE_CANDIDATE', 'false')
    monkeypatch.setenv('INSTANCE_CONFIRMATION_THRESHOLD', '5')
    monkeypatch.delenv('ENABLE_CONTRADICTION_CHECK', raising=False)
    monkeypatch.setenv('PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY', 'false')
    monkeypatch.setenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', 'false')
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'true')
    monkeypatch.setenv('INSTANCE_CONFIRMATION_MODE', 'contradiction')
    agent = SimpleNamespace(
        dataset='instancenav', obj_goal='chair', obj_goal_sg='chair', total_steps=0,
        goal_attributes={'intrinsic_attributes': 'A beige chair.'},
        scenegraph=SimpleNamespace(), current_obj_predictions=Predictions([[270, 220, 330, 280], [470, 220, 530, 280]]),
        depth=np.full((480, 640, 1), 3., dtype=np.float32),
        map_size_cm=2000, map_resolution=5, resolution=5, map_size=400,
        goal_gps_map=np.zeros((400, 400)), goal_gps_timestamp_map=np.zeros((400, 400), dtype=int),
        goal_map=np.zeros((400, 400)), goal_merge_threshold=.8, distance_threshold=4,
        found_goal=False, found_possible_goal=False, found_goal_times=0,
        goal_gps=np.zeros(2), possible_goal_temp_gps=np.zeros(2),
        rag_check_enabled=False, rag_sliding_window=[], rag_sliding_window_size=5,
        rag_latest_goal_detection=None, last_segment_num=0,
        get_goal_detection_threshold=lambda: 5,
    )

    def center(box, index, depth):
        point = ((box[:2] + box[2:]) // 2).to(torch.int64)
        return point, False, depth[point[1], point[0], 0]

    agent.get_unoccluded_center_point = center
    agent.get_goal_gps = lambda obs, angle, distance: np.array([
        np.cos(float(angle) / 180 * np.pi) * float(distance),
        np.sin(float(angle) / 180 * np.pi) * float(distance)])
    manager = ObjectDetectionManager(agent)
    manager.reset_instance_verification(episode())
    observations = {'rgb': np.zeros((480, 640, 3), dtype=np.uint8)}
    state = {'answers': ['unknown', 'unknown'], 'calls': 0, 'prompt': ''}

    def complete(prompt, image, **options):
        state['calls'] += 1
        state['prompt'] = prompt
        response = json.dumps({'matches': state['answers']})
        options['validator'](response)
        return response

    agent.scenegraph.model_client = SimpleNamespace(complete=complete)
    return agent, manager, observations, state


def verify_frame(agent, manager, observations):
    return manager._verify_instance_boxes(observations, agent.current_obj_predictions.bbox.tolist(),
                                          prediction_indices=list(range(len(agent.current_obj_predictions.bbox))))


def test_unknown_passes_only_in_new_mode_and_legacy_mode_does_not_load_csv(navigation, monkeypatch, capsys):
    agent, manager, observations, state = navigation
    assert verify_frame(agent, manager, observations) == [0, 1]
    assert 'significant conflict' in state['prompt']
    assert 'matches the intrinsic attributes' not in state['prompt']
    assert state['calls'] == 1 and not agent.instance_rejections
    output = capsys.readouterr().out
    records = [json.loads(line[len('[Instance Verification] '):])
               for line in output.splitlines() if line.startswith('[Instance Verification] ')]
    assert records[-1]['event'] == 'filter_result' and records[-1]['kept_box_indices'] == [0, 1]
    assert records[-1]['rejected_box_indices'] == []
    assert any(line.startswith('[Contradiction Check] ') and 'decision=unknown' in line
               for line in output.splitlines())
    monkeypatch.setenv('INSTANCE_CONFIRMATION_MODE', 'positive_match')
    manager.reset_instance_verification(SimpleNamespace())  # No factor lookup in legacy mode.
    assert not agent.goal_constraints
    agent.scenegraph.model_client.complete = lambda *a, **k: '{"matches":["unknown","yes"]}'
    assert verify_frame(agent, manager, observations) == [1]
    records = verification_records(capsys)
    assert [r['event'] for r in records] == ['raw_response', 'result', 'filter_result']
    assert records[0]['answer'] == '{"matches":["unknown","yes"]}'
    assert records[-1]['kept_box_indices'] == [1] and records[-1]['rejected_box_indices'] == [0]
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'false')
    agent.scenegraph.model_client.complete = lambda *a, **k: pytest.fail('Master switch must bypass model')
    assert verify_frame(agent, manager, observations) == [0, 1]
    assert verification_records(capsys) == []


def test_historical_veto_clears_only_nearby_counts_and_active_target(navigation, capsys):
    agent, manager, observations, state = navigation
    _, _, _, gps = manager._navigation_goal_geometry(agent.current_obj_predictions.bbox[0], 0, observations)
    x, y = int(200 - gps[1] * 20), int(200 + gps[0] * 20)
    agent.goal_gps_map[y, x] = 5
    agent.goal_gps_map[50, 50] = 2
    agent.goal_gps_timestamp_map[y, x] = 10
    agent.found_possible_goal = True
    agent.possible_goal_temp_gps = gps.copy()
    agent.goal_map[y, x] = 1
    state['answers'] = ['yes', 'no']
    assert verify_frame(agent, manager, observations) == [1]
    assert not agent.found_possible_goal and not agent.goal_map.any()
    assert agent.goal_gps_map[y, x] == 0 and agent.goal_gps_timestamp_map[y, x] == 0
    assert agent.goal_gps_map[50, 50] == 2
    assert agent.instance_rejections[0]['decision'] == 'yes'
    verification_records(capsys)
    # A nearby re-detection stays rejected; the remaining box is renumbered for the model.
    agent.current_obj_predictions.bbox[0] += torch.tensor([2, 0, 2, 0])
    state['answers'] = ['no']
    assert verify_frame(agent, manager, observations) == [1]
    assert state['calls'] == 2
    records = verification_records(capsys)
    assert [r['event'] for r in records] == ['history_skip', 'raw_response', 'result', 'filter_result']
    assert records[0]['skipped_box_indices'] == [0]
    assert records[1]['box_indices'] == [1]  # Model candidate 0 maps to original box 1.
    assert records[-1]['checked_box_indices'] == [1]
    assert records[-1]['kept_box_indices'] == [1] and records[-1]['rejected_box_indices'] == [0]
    assert all(r['verification_id'] for r in records)
    # Clearing temporary navigation state does not clear the semantic veto.
    agent.goal_gps_map.fill(0)
    agent.goal_gps_timestamp_map.fill(0)
    assert verify_frame(agent, manager, observations) == [1]
    manager.reset_instance_verification(episode())
    state['answers'] = ['unknown', 'unknown']
    assert verify_frame(agent, manager, observations) == [0, 1]


def test_a_conflict_in_second_box_overrides_same_position_first_box(navigation, capsys):
    agent, manager, observations, state = navigation
    agent.current_obj_predictions.bbox[1] = agent.current_obj_predictions.bbox[0]
    state['answers'] = ['no', 'yes']
    assert verify_frame(agent, manager, observations) == []
    assert state['calls'] == 1
    records = verification_records(capsys)
    assert records[-1]['contradicted_box_indices'] == [1]
    assert records[-1]['kept_box_indices'] == [] and records[-1]['rejected_box_indices'] == [0, 1]
    assert verify_frame(agent, manager, observations) == []
    assert state['calls'] == 1  # Known rejected candidates need no more API calls.
    records = verification_records(capsys)
    assert [r['event'] for r in records] == ['history_skip', 'filter_result']
    assert records[0]['skipped_box_indices'] == [0, 1]
    assert records[1]['checked_box_indices'] == [] and records[1]['kept_box_indices'] == []


def test_invalid_batch_or_api_error_cannot_mutate_candidate_history(navigation, capsys):
    agent, manager, observations, state = navigation
    state['answers'] = ['yes']  # One decision cannot cover two boxes.
    agent.goal_gps_map[200, 200] = 3
    before = agent.goal_gps_map.copy()
    with pytest.raises(ModelRequestError, match='3 format attempts'):
        verify_frame(agent, manager, observations)
    assert state['calls'] == 3 and not agent.instance_rejections
    assert np.array_equal(before, agent.goal_gps_map)
    records = verification_records(capsys)
    assert [r['event'] for r in records] == ['raw_response', 'format_error'] * 3 + ['failed']

    def fail(*args, **kwargs):
        raise ModelRequestError('API unavailable')

    agent.scenegraph.model_client.complete = fail
    with pytest.raises(ModelRequestError, match='API unavailable'):
        verify_frame(agent, manager, observations)
    assert not agent.instance_rejections
    records = verification_records(capsys)
    assert len(records) == 1 and records[0]['event'] == 'request_error'
    assert records[0]['error'] == 'ModelRequestError: API unavailable'


@pytest.mark.parametrize('threshold', [2, 4, 5])
def test_navigation_detector_keeps_existing_distance_and_count_rules(navigation, threshold, monkeypatch):
    monkeypatch.setenv("INSTANCE_CONFIRMATION_THRESHOLD", str(threshold))
    agent, manager, observations, state = navigation
    agent.current_obj_predictions = Predictions([[270, 220, 330, 280]])
    state['answers'] = ['unknown']
    agent.get_goal_detection_threshold = lambda: threshold
    agent.depth.fill(4.)
    manager._detect_goal_with_navigation_detector(observations)
    assert agent.found_possible_goal and not agent.found_goal and not agent.goal_gps_map.any()
    agent.depth.fill(3.)
    for i in range(threshold):
        agent.total_steps += 1
        manager._detect_goal_with_navigation_detector(observations)
        assert agent.found_goal == (i + 1 >= threshold)
    assert agent.goal_gps_map.max() == threshold


def test_far_candidate_can_be_rejected_before_entering_four_meters(navigation, monkeypatch):
    monkeypatch.setenv("VERIFY_BEFORE_CANDIDATE", "true")
    agent, manager, observations, state = navigation
    agent.current_obj_predictions = Predictions([[270, 220, 330, 280]])
    agent.depth.fill(6.)
    state['answers'] = ['unknown']
    manager._detect_goal_with_navigation_detector(observations)
    assert agent.found_possible_goal and not agent.goal_gps_map.any()
    state['answers'] = ['yes']
    manager._detect_goal_with_navigation_detector(observations)
    assert not agent.found_possible_goal and not agent.found_goal
    assert len(agent.instance_rejections) == 1


def test_scenegraph_path_uses_threshold_one_and_node_veto(navigation, monkeypatch):
    monkeypatch.setenv("INSTANCE_CONFIRMATION_THRESHOLD", "1")
    agent, manager, observations, state = navigation
    mask = np.zeros((480, 640), dtype=bool)
    mask[220:280, 270:330] = True
    node = SimpleNamespace(is_goal_node=True, object={'id': 7, 'image_idx': [0], 'mask_idx': [0]})
    agent.scenegraph.nodes = [node]
    agent.scenegraph.segment2d_results = [{'caption': ['chair'], 'mask': [mask]}]
    state['answers'] = ['unknown']
    manager._detect_goal_with_scenegraph(observations)
    assert agent.found_goal  # The shared threshold is explicitly one for this test.
    agent.found_goal = False
    state['answers'] = ['yes']
    assert manager._verify_instance_boxes(observations, [[270, 220, 330, 280]], masks=[mask], nodes=[node]) == []
    state['answers'] = ['no']
    before = state['calls']
    agent.depth.fill(5.)  # Existing node ID associates it even when the depth estimate changes.
    assert manager._verify_instance_boxes(observations, [[270, 220, 330, 280]], masks=[mask], nodes=[node]) == []
    assert state['calls'] == before
    other_node = SimpleNamespace(object={'id': 8})
    assert manager._verify_instance_boxes(observations, [[270, 220, 330, 280]], masks=[mask], nodes=[other_node]) == [0]


def test_switch_defaults_and_incompatible_configuration(monkeypatch):
    monkeypatch.delenv('ENABLE_CONTRADICTION_CHECK', raising=False)
    monkeypatch.delenv('INSTANCE_CONFIRMATION_MODE', raising=False)
    monkeypatch.delenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', raising=False)
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'true')
    assert instance_confirmation_mode() == 'positive_match'
    assert not post_confirmation_contradiction_check_enabled()
    monkeypatch.setenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', 'yes')
    assert post_confirmation_contradiction_check_enabled()
    monkeypatch.setenv('INSTANCE_CONFIRMATION_MODE', 'contradiction')
    with pytest.raises(ValueError, match='requires'):
        post_confirmation_contradiction_check_enabled()
    monkeypatch.setenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', 'false')
    assert instance_confirmation_mode() == 'contradiction'
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'false')
    assert instance_confirmation_mode() == 'contradiction'
    monkeypatch.setenv('INSTANCE_CONFIRMATION_MODE', 'positive_match')
    assert not post_confirmation_contradiction_check_enabled()
    monkeypatch.setenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', 'true')
    assert not post_confirmation_contradiction_check_enabled()
    monkeypatch.setenv('ENABLE_CONTRADICTION_CHECK', 'false')
    with pytest.raises(ValueError, match='retired'):
        instance_confirmation_mode()


def test_launcher_normalizes_overrides_and_exports_new_switch(tmp_path):
    fake_python = tmp_path / 'python'
    capture = tmp_path / 'capture'
    fake_python.write_text(
        '#!/bin/bash\nprintf "%s %s %s %s\\n" "$INSTANCE_CONFIRMATION_MODE" '
        '"$ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK" '
        '"$ENABLE_INSTANCE_CONFIRMATION" '
        '"$PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY" >> "$SWITCH_TEST_CAPTURE"\n'
        'if [[ "$1" == "-" ]]; then cat >/dev/null; fi\n')
    fake_python.chmod(0o755)
    env = {**os.environ, 'PSGNAV_PYTHON': str(fake_python), 'SWITCH_TEST_CAPTURE': str(capture),
           'INSTANCE_CONFIRMATION_MODE': 'contradiction',
           'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK': 'false',
           'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY': 'false',
           'ENABLE_INSTANCE_CONFIRMATION': 'true'}
    env.pop('ENABLE_CONTRADICTION_CHECK', None)
    command = ['bash', 'scripts/run_instance100.sh', '--check-only',
               '--instance-confirmation-mode', 'POSITIVE_MATCH',
               '--enable-post-confirmation-contradiction-check', 'YES',
               '--preserve-positive-evidence-on-recovery', 'YES']
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == ['positive_match true true true'] * 2
    assert 'INSTANCE_CONFIRMATION_MODE: positive_match' in result.stdout
    assert 'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK: true' in result.stdout
    assert 'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY: true' in result.stdout
    for options in [
        ['--instance-confirmation-mode', 'bad'],
        ['--instance-confirmation-mode'],
        ['--enable-post-confirmation-contradiction-check', 'bad'],
        ['--enable-post-confirmation-contradiction-check'],
        ['--enable-post-confirmation-contradiction-check', 'true'],
        ['--enable-contradiction-check', 'true'],
        ['--preserve-positive-evidence-on-recovery', 'bad'],
        ['--preserve-positive-evidence-on-recovery'],
        ['--preserve-positive-evidence-on-recovery', 'true'],
    ]:
        result = subprocess.run(['bash', 'scripts/run_instance100.sh', *options], env=env, capture_output=True, text=True)
        assert result.returncode == 2, options
    retired = subprocess.run(['bash', 'scripts/run_instance100.sh', '--check-only'],
                             env={**env, 'ENABLE_CONTRADICTION_CHECK': 'false'}, capture_output=True, text=True)
    assert retired.returncode == 2 and 'retired' in retired.stderr
    assert len(capture.read_text().splitlines()) == 2  # Invalid flags never start Python.


def test_launcher_selects_vlm_backend(tmp_path):
    fake_python = tmp_path / 'python'
    capture = tmp_path / 'capture'
    fake_python.write_text(
        '#!/bin/bash\nprintf "%s %s %s\\n" "$VLM" "$QWEN_MODEL_NAME" "$PSGNAV_MODEL_BACKEND" >> "$SWITCH_TEST_CAPTURE"\n'
        'if [[ "$1" == "-" ]]; then cat >/dev/null; fi\n')
    fake_python.chmod(0o755)
    env = {**os.environ, 'PSGNAV_PYTHON': str(fake_python), 'SWITCH_TEST_CAPTURE': str(capture)}
    env.pop('VLM', None)
    env.pop('QWEN_MODEL_NAME', None)
    env.pop('PSGNAV_MODEL_BACKEND', None)
    default = subprocess.run(['bash', 'scripts/run_instance100.sh', '--check-only'],
                             env=env, capture_output=True, text=True)
    assert default.returncode == 0, default.stderr
    assert capture.read_text().splitlines() == ['qwen3.5-flash qwen3.5-flash openai'] * 2
    assert 'VLM: qwen3.5-flash' in default.stdout
    assert 'PSGNAV_MODEL_BACKEND: openai' in default.stdout
    capture.write_text('')
    selected = subprocess.run(
        ['bash', 'scripts/run_instance100.sh', '--check-only', '--vlm', 'gemini-2.5-flash'],
        env=env, capture_output=True, text=True)
    assert selected.returncode == 0, selected.stderr
    assert capture.read_text().splitlines() == ['gemini-2.5-flash google/gemini-2.5-flash openrouter'] * 2
    assert 'QWEN_MODEL_NAME: google/gemini-2.5-flash' in selected.stdout
    assert 'PSGNAV_MODEL_BACKEND: openrouter' in selected.stdout
    capture.write_text('')
    silicon = subprocess.run(
        ['bash', 'scripts/run_instance100.sh', '--check-only', '--vlm', 'qwen3-vl-8b-instruct'],
        env=env, capture_output=True, text=True)
    assert silicon.returncode == 0, silicon.stderr
    assert capture.read_text().splitlines() == ['qwen3-vl-8b-instruct Qwen/Qwen3-VL-8B-Instruct siliconflow'] * 2
    assert 'PSGNAV_MODEL_BACKEND: siliconflow' in silicon.stdout
    for options in [['--vlm'], ['--vlm', 'qwen3-vl-flash'], ['--vlm', 'Qwen/Qwen3-VL-8B-Instruct']]:
        result = subprocess.run(['bash', 'scripts/run_instance100.sh', *options],
                                env=env, capture_output=True, text=True)
        assert result.returncode == 2, options


@pytest.fixture
def post_navigation(navigation, monkeypatch):
    agent, manager, observations, state = navigation
    monkeypatch.setenv('INSTANCE_CONFIRMATION_MODE', 'positive_match')
    monkeypatch.setenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', 'true')
    monkeypatch.delenv('PSGNAV_LOG_FILE', raising=False)
    monkeypatch.setattr('utils.instance_verifier.time.sleep', lambda _: None)
    manager.reset_instance_verification(episode())
    agent.current_obj_predictions = Predictions([[270, 220, 330, 280]])
    state.update(answers=['yes'], rechecks=[], decision='no_conflict', primary_complete=agent.scenegraph.model_client.complete)

    def complete(prompt, image=None, *, images=None, **options):
        if images is None:
            return state['primary_complete'](prompt, image, **options)
        assert not agent.found_goal  # The final state must never precede this gate.
        state['rechecks'].append([frame.copy() for frame in images])
        if state.get('error'):
            raise ModelRequestError(state['error'])
        decision = state['decision']
        response = json.dumps({'decision': decision, 'image_index': 0 if decision == 'conflict' else -1,
                               'reason': 'The target has arms; this chair is armless.' if decision == 'conflict' else ''})
        options['validator'](response)
        return response

    agent.scenegraph.model_client = SimpleNamespace(complete=complete)
    return agent, manager, observations, state


def observe_navigation(fixture, step):
    agent, manager, observations, _ = fixture
    agent.total_steps = step
    observations['rgb'].fill(step)
    manager._detect_goal_with_navigation_detector(observations)


@pytest.mark.parametrize('decision', ['no_conflict', 'unknown'])
def test_post_check_runs_only_at_final_count_with_all_positive_images(post_navigation, decision):
    agent, manager, observations, state = post_navigation
    state['decision'] = decision
    for step in range(4):
        observe_navigation(post_navigation, step)
    assert not agent.found_goal and not state['rechecks']
    assert agent.goal_gps_map.max() == 4
    observe_navigation(post_navigation, 4)
    assert agent.found_goal and len(state['rechecks']) == 1
    assert [frame.getpixel((0, 0))[0] for frame in state['rechecks'][0]] == list(range(5))
    assert all(frame.getpixel((270, 220)) == (255, 0, 0) for frame in state['rechecks'][0])
    assert len(manager._positive_evidence.tracks) == 1


def test_post_deduplicates_images_without_changing_repeated_detection_counts(post_navigation):
    agent, manager, observations, state = post_navigation
    for step in (0, 1, 2, 3, 3):
        observe_navigation(post_navigation, step)
    assert agent.found_goal and agent.goal_gps_map.max() == 5
    assert [frame.getpixel((0, 0))[0] for frame in state['rechecks'][0]] == [0, 1, 2, 3]


def test_post_conflict_clears_votes_but_keeps_evidence_and_allows_reconsideration(post_navigation):
    agent, manager, observations, state = post_navigation
    state['decision'] = 'conflict'
    for step in range(5):
        observe_navigation(post_navigation, step)
    assert not agent.found_goal and not agent.found_possible_goal
    assert not agent.goal_gps_map.any() and not agent.instance_rejections
    assert len(state['rechecks']) == 1
    observe_navigation(post_navigation, 4)
    assert not agent.goal_gps_map.any() and len(state['rechecks']) == 1
    state['decision'] = 'no_conflict'
    for step in range(5, 10):
        observe_navigation(post_navigation, step)
    assert agent.found_goal and len(state['rechecks']) == 2
    # No last-K window, and the earlier rejected batch remains available for review.
    assert [frame.getpixel((0, 0))[0] for frame in state['rechecks'][-1]] == list(range(10))


def test_post_api_failure_never_becomes_a_confirmed_goal(post_navigation, capsys):
    agent, manager, observations, state = post_navigation
    state['error'] = 'Provider unavailable'
    for step in range(4):
        observe_navigation(post_navigation, step)
    with pytest.raises(ModelRequestError, match='failed after 3 attempts'):
        observe_navigation(post_navigation, 4)
    assert not agent.found_goal and len(state['rechecks']) == 3
    assert 'Provider unavailable' in capsys.readouterr().err


def test_post_neighbor_images_and_confirmation_votes_stay_separate(post_navigation):
    agent, manager, observations, state = post_navigation
    agent.current_obj_predictions = Predictions([[270, 220, 330, 280], [335, 220, 395, 280]])
    state['answers'] = ['yes', 'yes']
    for step in range(2):
        observe_navigation(post_navigation, step)
    assert not agent.found_goal and not state['rechecks']
    assert len(manager._positive_evidence.tracks) == 2
    assert [len(track.votes) for track in manager._positive_evidence.tracks] == [2, 2]
    observe_navigation(post_navigation, 2)
    assert not agent.found_goal and agent.goal_gps_map.max() == 6
    assert not state['rechecks']
    observe_navigation(post_navigation, 3)
    observe_navigation(post_navigation, 4)
    assert agent.found_goal
    assert len(state['rechecks'][0]) == 5
    assert state['rechecks'][0][0].getpixel((335, 220)) != (255, 0, 0)


def test_post_veto_removes_only_that_candidates_contributions_to_shared_cell(post_navigation):
    agent, manager, observations, state = post_navigation
    agent.current_obj_predictions = Predictions([[270, 220, 330, 280], [335, 220, 395, 280]])
    state.update(answers=['yes', 'yes'], decision='conflict')
    for step in range(4):
        observe_navigation(post_navigation, step)
    # Only the left candidate reaches five; its veto must leave the right's four.
    agent.current_obj_predictions = Predictions([[270, 220, 330, 280]])
    state['answers'] = ['yes']
    observe_navigation(post_navigation, 4)
    assert not agent.found_goal
    assert agent.goal_gps_map.max() == 4
    assert [len(track.votes) for track in manager._positive_evidence.tracks] == [0, 4]
    assert len(state['rechecks']) == 1 and not agent.instance_rejections


@pytest.mark.parametrize('boxes,steps', [
    ([[270, 220, 330, 280]], [0, 1, 2, 3, 3]),
    ([[270, 220, 330, 280], [335, 220, 395, 280]], [0, 1, 2]),
])
def test_passing_post_check_preserves_baseline_counts_timing_and_goal_position(post_navigation, monkeypatch, boxes, steps):
    agent, manager, observations, state = post_navigation
    agent.current_obj_predictions = Predictions(boxes)
    state['answers'] = ['yes'] * len(boxes)
    baseline = copy.deepcopy(agent)
    monkeypatch.setenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', 'false')
    baseline_manager = ObjectDetectionManager(baseline)
    baseline_manager.reset_instance_verification(episode())
    expected = []
    for step in steps:
        baseline.total_steps = step
        baseline_manager._detect_goal_with_navigation_detector(observations)
        expected.append((baseline.found_goal, baseline.goal_gps_map.copy(), baseline.goal_gps.copy()))
    monkeypatch.setenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', 'true')
    for step, (found_goal, count_map, gps) in zip(steps, expected):
        observe_navigation(post_navigation, step)
        assert agent.found_goal == found_goal
        assert np.array_equal(agent.goal_gps_map, count_map)
        if found_goal:
            assert np.array_equal(agent.goal_gps, gps)


@pytest.mark.parametrize('distance', [float('nan'), 0., 100.])
def test_post_unlocalized_candidate_never_confirms(post_navigation, distance):
    agent, manager, observations, state = post_navigation
    agent.depth.fill(distance)
    for step in range(5):
        observe_navigation(post_navigation, step)
    assert not agent.found_goal and not state['rechecks']


def test_post_recovery_preserves_all_views_and_episode_reset_drops_them(post_navigation):
    agent, manager, observations, state = post_navigation
    observe_navigation(post_navigation, 0)
    observe_navigation(post_navigation, 1)
    manager.reset_post_confirmation_progress()
    agent.goal_gps_map.fill(0)
    for step in range(2, 7):
        observe_navigation(post_navigation, step)
    assert agent.found_goal and len(state['rechecks'][0]) == 7
    manager.reset_instance_verification(episode())
    assert not manager._positive_evidence.tracks and not manager._post_frame_tracks


def test_post_scenegraph_rechecks_before_direct_lock_and_retains_far_views(post_navigation, monkeypatch):
    monkeypatch.setenv("INSTANCE_CONFIRMATION_THRESHOLD", "1")
    monkeypatch.setenv("VERIFY_BEFORE_CANDIDATE", "true")
    agent, manager, observations, state = post_navigation
    mask = np.zeros((480, 640), dtype=bool)
    mask[220:280, 270:330] = True
    node = SimpleNamespace(is_goal_node=True, object={'id': 7, 'image_idx': [], 'mask_idx': []})
    agent.scenegraph.nodes = [node]
    agent.scenegraph.segment2d_results = []

    def observe(step, distance):
        agent.total_steps = step
        agent.depth.fill(distance)
        observations['rgb'].fill(step)
        node.object['image_idx'].append(step)
        node.object['mask_idx'].append(0)
        agent.scenegraph.segment2d_results.append({'caption': ['chair'], 'mask': [mask]})
        manager._detect_goal_with_scenegraph(observations)

    observe(0, 5.)
    assert agent.found_possible_goal and not state['rechecks']
    state['decision'] = 'conflict'
    observe(1, 3.)
    assert not agent.found_goal and not agent.found_possible_goal
    assert len(state['rechecks'][0]) == 2 and not agent.instance_rejections
    state['decision'] = 'unknown'
    observe(2, 3.)
    assert agent.found_goal and len(state['rechecks'][-1]) == 3


def test_recovery_switch_defaults_and_constraints(monkeypatch):
    monkeypatch.delenv('PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY', raising=False)
    monkeypatch.delenv('ENABLE_CONTRADICTION_CHECK', raising=False)
    monkeypatch.setenv('INSTANCE_CONFIRMATION_MODE', 'positive_match')
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'true')
    assert not preserve_positive_evidence_on_recovery_enabled()
    monkeypatch.setenv('PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY', 'YES')
    assert preserve_positive_evidence_on_recovery_enabled()
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'false')
    assert not preserve_positive_evidence_on_recovery_enabled()
    monkeypatch.setenv('ENABLE_INSTANCE_CONFIRMATION', 'true')
    monkeypatch.setenv('INSTANCE_CONFIRMATION_MODE', 'contradiction')
    with pytest.raises(ValueError, match='requires'):
        preserve_positive_evidence_on_recovery_enabled()
    monkeypatch.setenv('PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY', 'sometimes')
    with pytest.raises(ValueError, match='true/false'):
        preserve_positive_evidence_on_recovery_enabled()


@pytest.fixture(params=[False, True], ids=['without_post_check', 'with_post_check'])
def recovery_navigation(post_navigation, monkeypatch, request):
    PSG_Nav_Agent = load_agent_methods('reset_goal_detection')
    agent, manager, _, state = post_navigation
    monkeypatch.setenv('PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY', 'true')
    monkeypatch.setenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', str(request.param).lower())
    manager.reset_instance_verification(episode())
    agent.detection_manager = manager
    agent.reset_goal_detection = lambda reason='unspecified': PSG_Nav_Agent.reset_goal_detection(agent, reason)
    agent.last_gps = np.zeros(2)
    agent.in_stuck_recovery = False
    agent.not_move_steps = agent.forward_no_move_count = 0
    state['post_enabled'] = request.param
    return post_navigation


def test_recovery_keeps_three_votes_then_continues_to_four_and_five(recovery_navigation):
    agent, manager, _, state = recovery_navigation
    for step in (1, 2, 3):
        observe_navigation(recovery_navigation, step)
    old_map = agent.goal_gps_map.copy()
    old_timestamps = agent.goal_gps_timestamp_map.copy()
    for step in (4, 5):
        agent.total_steps = step
        agent.reset_goal_detection(reason='stuck')
        assert not agent.found_goal and not agent.found_possible_goal and not agent.goal_map.any()
        assert len(manager._positive_evidence.tracks[0].votes) == 3
        assert np.array_equal(agent.goal_gps_map, old_map)
        assert np.array_equal(agent.goal_gps_timestamp_map, old_timestamps)
    agent.last_gps = np.array([.1, 0.])
    observe_navigation(recovery_navigation, 6)
    assert agent.goal_gps_map.max() == 4 and not agent.found_goal
    observe_navigation(recovery_navigation, 7)
    assert agent.found_goal and agent.found_goal_times == 5
    track = manager._positive_evidence.tracks[0]
    if state['post_enabled']:
        assert [frame.getpixel((0, 0))[0] for frame in state['rechecks'][0]] == [1, 2, 3, 6, 7]
    else:
        assert not track.frames and not state['rechecks']


def test_recovery_requires_real_progress_and_uses_new_position(recovery_navigation):
    agent, manager, _, state = recovery_navigation
    for step in range(1, 6):
        observe_navigation(recovery_navigation, step)
    assert agent.found_goal
    old_goal = agent.goal_gps.copy()
    agent.reset_goal_detection(reason='stuck')
    observe_navigation(recovery_navigation, 6)
    assert not agent.found_goal and not agent.found_possible_goal  # Turning/completion message isn't movement.
    assert len(manager._positive_evidence.tracks[0].votes) == 6
    agent.last_gps = np.array([.1, 0.])
    agent.in_stuck_recovery = True
    observe_navigation(recovery_navigation, 7)
    assert not agent.found_goal
    agent.in_stuck_recovery = False
    agent.depth.fill(3.2)
    observe_navigation(recovery_navigation, 8)
    assert agent.found_goal
    assert not np.array_equal(old_goal, agent.goal_gps)
    assert np.array_equal(agent.goal_gps, manager._positive_evidence.tracks[0].gps)
    assert len(state['rechecks']) == (2 if state['post_enabled'] else 0)


def test_same_failed_approach_is_deferred_without_losing_votes(recovery_navigation):
    agent, manager, _, state = recovery_navigation
    for step in range(1, 6):
        observe_navigation(recovery_navigation, step)
    agent.reset_goal_detection(reason='stuck')
    agent.last_gps = np.array([.1, 0.])
    observe_navigation(recovery_navigation, 6)
    assert agent.found_goal
    agent.reset_goal_detection(reason='stuck')  # Second failure within .25 m of the first.
    agent.last_gps = np.array([.2, 0.])
    observe_navigation(recovery_navigation, 7)
    assert not agent.found_goal and not agent.found_possible_goal
    assert len(manager._positive_evidence.tracks[0].votes) == 7
    agent.last_gps = np.array([.7, 0.])
    observe_navigation(recovery_navigation, 8)
    assert agent.found_goal and agent.found_goal_times == 8


def test_nearby_candidate_cannot_borrow_retained_votes(recovery_navigation):
    agent, manager, _, state = recovery_navigation
    # These two boxes deliberately share the legacy spatial counter cell.
    agent.current_obj_predictions = Predictions([[270, 220, 330, 280], [335, 220, 395, 280]])
    state['answers'] = ['yes', 'yes']
    for step in range(1, 4):
        observe_navigation(recovery_navigation, step)
    assert agent.goal_gps_map.max() == 6
    assert not agent.found_goal  # Neither candidate has five votes of its own.
    assert [len(t.votes) for t in manager._positive_evidence.tracks] == [3, 3]
    agent.reset_goal_detection(reason='stuck')
    agent.last_gps = np.array([.1, 0.])
    agent.current_obj_predictions = Predictions([[335, 220, 395, 280]])
    state['answers'] = ['yes']
    observe_navigation(recovery_navigation, 4)
    assert not agent.found_goal
    assert [len(t.votes) for t in manager._positive_evidence.tracks] == [3, 4]
    observe_navigation(recovery_navigation, 5)
    assert agent.found_goal and manager._post_goal_track_id == 1


def test_new_candidate_cannot_inherit_old_count(recovery_navigation):
    agent, manager, _, state = recovery_navigation
    for step in (1, 2, 3):
        observe_navigation(recovery_navigation, step)
    agent.reset_goal_detection(reason='stuck')
    agent.last_gps = np.array([.1, 0.])
    agent.current_obj_predictions = Predictions([[470, 220, 530, 280]])
    observe_navigation(recovery_navigation, 4)
    assert [len(t.votes) for t in manager._positive_evidence.tracks] == [3, 1]
    assert not agent.found_goal


def test_explicit_rejection_and_episode_reset_do_not_resurrect_votes(recovery_navigation):
    agent, manager, _, state = recovery_navigation
    for step in (1, 2, 3):
        observe_navigation(recovery_navigation, step)
    agent.goal_gps_map[50, 50] = -1
    agent.reset_goal_detection(reason='rag_rejected')
    assert not manager._positive_evidence.tracks[0].votes
    assert agent.goal_gps_map.max() == 0 and agent.goal_gps_map[50, 50] == -1
    agent.reset_goal_detection(reason='stuck')
    agent.last_gps = np.array([.1, 0.])
    observe_navigation(recovery_navigation, 4)
    assert len(manager._positive_evidence.tracks[0].votes) == 1 and not agent.found_goal
    manager.reset_instance_verification(episode())
    assert not manager._positive_evidence.tracks and manager._recovery_guard.step == -1


def test_preservation_does_not_change_same_frame_counting(recovery_navigation):
    agent, manager, _, _ = recovery_navigation
    for step in (1, 2, 3, 4, 4):
        observe_navigation(recovery_navigation, step)
    assert agent.found_goal and len(manager._positive_evidence.tracks[0].votes) == 5


@pytest.mark.parametrize('post_enabled', [False, True])
def test_disabled_preservation_still_clears_counts(post_navigation, monkeypatch, post_enabled):
    PSG_Nav_Agent = load_agent_methods('reset_goal_detection')
    agent, manager, _, _ = post_navigation
    monkeypatch.setenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', str(post_enabled).lower())
    manager.reset_instance_verification(episode())
    agent.detection_manager = manager
    for step in (1, 2, 3):
        observe_navigation(post_navigation, step)
    PSG_Nav_Agent.reset_goal_detection(agent, reason='stuck')
    assert not agent.goal_gps_map.any()
    if post_enabled:
        assert not manager._positive_evidence.tracks[0].votes
    observe_navigation(post_navigation, 4)
    assert agent.goal_gps_map.max() == 1 and not agent.found_goal


def test_ambiguous_reassociation_keeps_old_records_without_inheriting(recovery_navigation):
    agent, manager, _, state = recovery_navigation
    agent.current_obj_predictions = Predictions([[270, 220, 330, 280], [335, 220, 395, 280]])
    state['answers'] = ['yes', 'yes']
    for step in (1, 2, 3):
        observe_navigation(recovery_navigation, step)
    agent.reset_goal_detection(reason='stuck')
    agent.last_gps = np.array([.1, 0.])
    # Midway between the previous estimates: neither association is reliable.
    agent.current_obj_predictions = Predictions([[302, 220, 362, 280]])
    state['answers'] = ['yes']
    observe_navigation(recovery_navigation, 4)
    assert [len(t.votes) for t in manager._positive_evidence.tracks] == [3, 3, 1]
    assert not agent.found_goal


def test_post_veto_survives_a_later_preserving_recovery(recovery_navigation):
    agent, manager, _, state = recovery_navigation
    if not state['post_enabled']:
        return
    state['decision'] = 'conflict'
    for step in range(1, 6):
        observe_navigation(recovery_navigation, step)
    assert not agent.found_goal and not manager._positive_evidence.tracks[0].votes
    assert len(manager._positive_evidence.tracks[0].frames) == 5
    agent.reset_goal_detection(reason='stuck')
    agent.last_gps = np.array([.1, 0.])
    state['decision'] = 'no_conflict'
    observe_navigation(recovery_navigation, 6)
    assert len(manager._positive_evidence.tracks[0].votes) == 1 and not agent.found_goal
    assert len(state['rechecks']) == 1


def test_scenegraph_recovery_requires_fresh_match_and_movement(recovery_navigation, monkeypatch):
    monkeypatch.setenv("INSTANCE_CONFIRMATION_THRESHOLD", "1")
    agent, manager, observations, state = recovery_navigation
    mask = np.zeros((480, 640), dtype=bool)
    mask[220:280, 270:330] = True
    node = SimpleNamespace(is_goal_node=True, object={'id': 7, 'image_idx': [], 'mask_idx': []})
    agent.scenegraph.nodes = [node]
    agent.scenegraph.segment2d_results = []

    def observe(step):
        agent.total_steps = step
        observations['rgb'].fill(step)
        node.object['image_idx'].append(len(agent.scenegraph.segment2d_results))
        node.object['mask_idx'].append(0)
        agent.scenegraph.segment2d_results.append({'caption': ['chair'], 'mask': [mask]})
        manager._detect_goal_with_scenegraph(observations)

    observe(1)
    assert agent.found_goal
    agent.reset_goal_detection(reason='stuck')
    observe(2)
    assert not agent.found_goal and not agent.found_possible_goal
    agent.last_gps = np.array([.1, 0.])
    state['answers'] = ['no']
    observe(3)
    assert not agent.found_goal
    state['answers'] = ['yes']
    observe(4)
    assert agent.found_goal and len(manager._positive_evidence.tracks) == 1
    if state['post_enabled']:
        assert [f.getpixel((0, 0))[0] for f in state['rechecks'][-1]] == [1, 2, 4]


def test_far_candidate_cannot_bypass_recovery_guard(recovery_navigation):
    agent, manager, _, _ = recovery_navigation
    agent.depth.fill(3.8)
    for step in (1, 2, 3):
        observe_navigation(recovery_navigation, step)
    track = manager._positive_evidence.tracks[0]
    agent.reset_goal_detection(reason='stuck')
    agent.depth.fill(4.2)
    # An exploration goal must survive a blocked, far-away positive observation.
    agent.goal_map[50, 60] = 1
    observe_navigation(recovery_navigation, 4)
    assert not agent.found_goal and not agent.found_possible_goal
    assert np.array_equal(agent.possible_goal_temp_gps, np.zeros(2))
    assert agent.goal_map[50, 60] == 1
    assert len(manager._positive_evidence.tracks) == 1 and len(track.votes) == 3
    # Once progress is observed, approach uses the fresh candidate position.
    agent.last_gps = np.array([.1, 0.])
    observe_navigation(recovery_navigation, 5)
    assert agent.found_possible_goal and not agent.found_goal
    assert np.array_equal(agent.possible_goal_temp_gps, track.gps)
    assert manager._post_possible_track_id == track.track_id


@pytest.mark.parametrize('rejected_answer', ['no', 'unknown'])
@pytest.mark.parametrize('positive_first', [False, True])
def test_rejected_duplicate_cannot_claim_positive_track(recovery_navigation, rejected_answer, positive_first):
    agent, manager, _, state = recovery_navigation
    for step in (1, 2, 3):
        observe_navigation(recovery_navigation, step)
    old_box = [270, 220, 330, 280]
    positive_box = [276, 222, 338, 282]
    boxes, answers = [old_box, positive_box], [rejected_answer, 'yes']
    if positive_first:
        boxes.reverse()
        answers.reverse()
    agent.current_obj_predictions = Predictions(boxes)
    state['answers'] = answers
    observe_navigation(recovery_navigation, 4)
    assert len(manager._positive_evidence.tracks) == 1
    track = manager._positive_evidence.tracks[0]
    assert len(track.votes) == 4 and agent.goal_gps_map.max() == 4
    assert track.box == positive_box and not agent.found_goal
    assert manager._post_frame_tracks[answers.index(rejected_answer)] is None
    assert manager._post_frame_tracks[answers.index('yes')] is track
    observe_navigation(recovery_navigation, 5)
    assert agent.found_goal and agent.found_goal_times == 5


def test_negative_only_observation_does_not_move_or_create_positive_tracks(recovery_navigation):
    agent, manager, _, state = recovery_navigation
    observe_navigation(recovery_navigation, 1)
    track = manager._positive_evidence.tracks[0]
    old_gps, old_box = track.gps.copy(), track.box.copy()
    state['answers'] = ['no', 'unknown']
    agent.current_obj_predictions = Predictions([[276, 222, 338, 282], [470, 220, 530, 280]])
    observe_navigation(recovery_navigation, 2)
    assert len(manager._positive_evidence.tracks) == 1
    assert np.array_equal(track.gps, old_gps) and track.box == old_box
    assert track.step == 1 and len(track.votes) == 1
    assert manager._post_frame_tracks == [None, None]


@pytest.mark.parametrize('post_enabled', [False, True])
def test_far_candidate_without_preservation_keeps_normal_approach(post_navigation, monkeypatch, post_enabled):
    agent, manager, observations, _ = post_navigation
    monkeypatch.setenv('ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK', str(post_enabled).lower())
    manager.reset_instance_verification(episode())
    agent.depth.fill(5.)
    observe_navigation(post_navigation, 1)
    _, _, _, gps = manager._navigation_goal_geometry(agent.current_obj_predictions.bbox[0], 0, observations)
    assert agent.found_possible_goal and not agent.found_goal
    assert np.array_equal(agent.possible_goal_temp_gps, gps)
    assert not agent.goal_gps_map.any()
