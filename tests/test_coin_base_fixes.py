"""E2 base fixes for CoIN: goal refresh after lock, depth valid-fraction gate, new prompt."""
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent_test_support import load_agent_methods
from utils.object_detection import ObjectDetectionManager

ROOT = Path(__file__).resolve().parents[1]


class Predictions:
    def __init__(self, boxes, labels):
        self.bbox = torch.tensor(boxes)
        self.fields = {'labels': list(labels), 'scores': torch.ones(len(boxes)) * .9}

    def get_field(self, field):
        return self.fields[field]

    def add_field(self, field, value):
        self.fields[field] = value

    def __len__(self):
        return len(self.bbox)


def _depth_agent(depth):
    cls = load_agent_methods('get_unoccluded_center_point')
    agent = cls.__new__(cls)
    agent.current_obj_predictions = Predictions([[10, 10, 50, 50]], ['picture'])
    return agent, depth


def test_depth_gate_off_keeps_mean_of_borrowed_pixels(monkeypatch):
    monkeypatch.delenv('DEPTH_VALID_FRACTION_GATE', raising=False)
    depth = np.full((64, 64, 1), 100., dtype=np.float32)  # mirror: invalid marker
    depth[14:18, 14:46, 0] = 2.0  # a thin valid strip (frame) inside the centre
    agent, depth = _depth_agent(depth)
    _, _, distance = agent.get_unoccluded_center_point(torch.tensor([10, 10, 50, 50]), 0, depth)
    assert distance == pytest.approx(2.0)


def test_depth_gate_drops_mostly_invalid_box_and_uses_median(monkeypatch):
    monkeypatch.setenv('DEPTH_VALID_FRACTION_GATE', '0.5')
    depth = np.full((64, 64, 1), 100., dtype=np.float32)
    depth[14:18, 14:46, 0] = 2.0
    agent, mirror = _depth_agent(depth)
    _, _, distance = agent.get_unoccluded_center_point(torch.tensor([10, 10, 50, 50]), 0, mirror)
    assert distance >= 100
    depth = np.full((64, 64, 1), 3.0, dtype=np.float32)
    depth[14:16, 14:46, 0] = 0.4  # outliers move the mean but not the median
    agent, solid = _depth_agent(depth)
    _, _, distance = agent.get_unoccluded_center_point(torch.tensor([10, 10, 50, 50]), 0, solid)
    assert distance == pytest.approx(3.0)


def test_gated_box_gets_no_gps_and_no_fallback_search(monkeypatch):
    monkeypatch.setenv('DEPTH_VALID_FRACTION_GATE', '0.5')
    agent = SimpleNamespace(depth=np.full((64, 64, 1), 2., dtype=np.float32),
                            get_unoccluded_center_point=lambda b, i, d: (torch.tensor([30, 30]), False, 100.0),
                            get_goal_gps=lambda obs, a, d: np.array([float(d), 0.]))
    manager = ObjectDetectionManager(agent)
    _, _, distance, gps = manager._navigation_goal_geometry(torch.tensor([10, 10, 50, 50]), 0, {})
    assert distance >= 100 and gps is None
    monkeypatch.setenv('DEPTH_VALID_FRACTION_GATE', '0')
    _, _, distance, gps = manager._navigation_goal_geometry(torch.tensor([10, 10, 50, 50]), 0, {})
    assert distance == pytest.approx(2.0)  # original neighbouring-pixel fallback


@pytest.mark.parametrize('name,value', [('DEPTH_VALID_FRACTION_GATE', '1'), ('DEPTH_VALID_FRACTION_GATE', '-0.1'),
                                        ('GOAL_REFRESH_RADIUS_M', '0'), ('GOAL_REFRESH_RADIUS_M', '6')])
def test_new_numeric_switches_are_validated(monkeypatch, name, value):
    from configs.navigation_config import depth_valid_fraction_gate, goal_refresh_radius_m
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        (depth_valid_fraction_gate if name.startswith('DEPTH') else goal_refresh_radius_m)()


def _refresh_manager(monkeypatch, boxes, labels, distances):
    import utils.object_detection as od
    monkeypatch.setattr(od, 'DETECTOR_TYPE', 'glip')
    predictions = Predictions(boxes, labels)
    agent = SimpleNamespace(found_goal=True, obj_goal='picture', dataset='coin', total_steps=7,
                            goal_gps=np.array([2.0, 0.0]), distance_threshold=4,
                            current_obj_predictions=predictions)
    manager = ObjectDetectionManager(agent)
    monkeypatch.setattr(manager, '_detect_with_glip', lambda obs: None)
    geometry = {j: (None, 0., d, np.array([d, float(j)]) if d < 100 else None)
                for j, d in enumerate(distances)}
    monkeypatch.setattr(manager, '_navigation_goal_geometry', lambda box, j, obs: geometry[j])
    return agent, manager


def test_refresh_moves_goal_to_nearest_same_category_box_within_radius(monkeypatch, capsys):
    monkeypatch.setenv('GOAL_REFRESH_RADIUS_M', '1.5')
    # box 0: gps (2.5, 0) jump 0.5; box 1: gps (2.2, 1) jump ~1.02; box 2 other category.
    agent, manager = _refresh_manager(monkeypatch, [[0, 0, 5, 5]] * 3, ['picture', 'picture', 'chair'],
                                      [2.5, 2.2, 2.0])
    gps = manager.refresh_locked_goal({})
    assert gps.tolist() == [2.5, 0.0] and agent.goal_gps.tolist() == [2.5, 0.0]
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith('[Goal Refresh] '))
    record = json.loads(line[len('[Goal Refresh] '):])
    assert record['previous_gps'] == [2.0, 0.0] and record['jump_m'] == 0.5


def test_refresh_ignores_far_instances_and_invalid_depth(monkeypatch):
    monkeypatch.setenv('GOAL_REFRESH_RADIUS_M', '1.0')
    # box 0: gps (3.5, 0) jump 1.5 > radius (another instance); box 1: invalid depth.
    agent, manager = _refresh_manager(monkeypatch, [[0, 0, 5, 5]] * 2, ['picture', 'picture'], [3.5, 100.])
    assert manager.refresh_locked_goal({}) is None
    assert agent.goal_gps.tolist() == [2.0, 0.0]
    agent.found_goal = False
    assert manager.refresh_locked_goal({}) is None


def test_perception_refreshes_only_when_enabled_and_locked(monkeypatch):
    from graph.meta_func import nav_func
    calls = []
    agent = SimpleNamespace(found_goal=True,
                            detection_manager=SimpleNamespace(refresh_locked_goal=lambda obs: calls.append(obs)),
                            detect_objects=lambda obs: pytest.fail('locked agent must not verify again'))
    owner = SimpleNamespace(agent=agent, observations={'rgb': None})
    perception = next(v for k, v in vars(nav_func).items()
                      if isinstance(v, type) and 'perception' in vars(v)).perception
    monkeypatch.setenv('GOAL_REFRESH_AFTER_LOCK', 'false')
    perception(owner)
    assert calls == []
    monkeypatch.setenv('GOAL_REFRESH_AFTER_LOCK', 'true')
    perception(owner)
    assert calls == [owner.observations]


def test_context_optional_prompt_is_selectable_and_distinct(monkeypatch):
    from utils.instance_verifier import POSITIVE_MATCH_RULES, verify_candidate_decisions
    prompts = {}

    def complete(prompt, image, **options):
        prompts[os.environ['INSTANCE_VERIFICATION_PROMPT']] = prompt
        answer = '{"matches":["yes"]}'
        options['validator'](answer)
        return answer

    for name in ('match', 'match_strict_no', 'match_context_optional'):
        monkeypatch.setenv('INSTANCE_VERIFICATION_PROMPT', name)
        assert verify_candidate_decisions(SimpleNamespace(complete=complete), np.zeros((20, 20, 3), np.uint8),
                                          [[1, 1, 5, 5]], 'picture',
                                          {'intrinsic_attributes': 'a picture of a flag'}) == ['yes']
    assert len(set(prompts.values())) == 3
    assert POSITIVE_MATCH_RULES['match_context_optional'] in prompts['match_context_optional']
    assert 'outside this view do not count' in prompts['match_context_optional']


def _launcher_capture(tmp_path, script, *options):
    fake_python = tmp_path / 'python'
    capture = tmp_path / 'capture'
    fake_python.write_text(
        '#!/bin/bash\nprintf "%s|%s|%s|%s|%s\\n" "$ENABLE_CANDIDATE_APPROACH" "$INSTANCE_VERIFICATION_PROMPT" '
        '"$GOAL_REFRESH_AFTER_LOCK" "$GOAL_REFRESH_RADIUS_M" "$DEPTH_VALID_FRACTION_GATE" >> "$SWITCH_TEST_CAPTURE"\n'
        'if [[ "$1" == "-" ]]; then cat >/dev/null; fi\n')
    fake_python.chmod(0o755)
    env = {**os.environ, 'PSGNAV_PYTHON': str(fake_python), 'SWITCH_TEST_CAPTURE': str(capture)}
    for name in ('ENABLE_CANDIDATE_APPROACH', 'INSTANCE_VERIFICATION_PROMPT', 'GOAL_REFRESH_AFTER_LOCK',
                 'GOAL_REFRESH_RADIUS_M', 'DEPTH_VALID_FRACTION_GATE'):
        env.pop(name, None)
    result = subprocess.run(['bash', script, '--check-only', '--full', *options], env=env,
                            capture_output=True, text=True, cwd=ROOT)
    lines = capture.read_text().splitlines() if capture.exists() else []
    return result, lines[-1] if lines else None


def test_e0_e2_presets_export_their_switches(tmp_path):
    result, e0 = _launcher_capture(tmp_path, 'scripts/run_coin_e0.sh')
    assert result.returncode == 0, result.stderr
    assert e0 == 'false|match_strict_no|false|1.5|0'
    result, e0 = _launcher_capture(tmp_path, 'scripts/run_coin_e0.sh',
                                   '--instance-verification-prompt', 'match_visible_evidence')
    assert result.returncode == 0, result.stderr
    assert e0 == 'false|match_visible_evidence|false|1.5|0'
    result, e2 = _launcher_capture(tmp_path, 'scripts/run_coin_e2.sh')
    assert result.returncode == 0, result.stderr
    assert e2 == 'false|match_strict_no|true|1.5|0.5'
    assert 'GOAL_REFRESH_AFTER_LOCK: true' in result.stdout
    result, e2 = _launcher_capture(tmp_path, 'scripts/run_coin_e2.sh',
                                   '--instance-verification-prompt', 'match_context_optional')
    assert result.returncode == 0, result.stderr
    assert e2 == 'false|match_context_optional|true|1.5|0.5'


@pytest.mark.parametrize('options', [['--depth-valid-fraction-gate', '1'], ['--goal-refresh-radius-m', '0'],
                                     ['--goal-refresh-after-lock', 'maybe'],
                                     ['--instance-verification-prompt', 'strict']])
def test_launcher_rejects_bad_new_switches(tmp_path, options):
    result, captured = _launcher_capture(tmp_path, 'scripts/run_coin.sh', *options)
    assert result.returncode == 2 and captured is None


SCHEMA_ECHO = ('[{"type": "object", "properties": {"matches": {"type": "array", "items": {"type": "string", '
               '"enum": ["yes", "no", "unknown"]}, "minItems": 1, "maxItems": 1}}, "required": ["matches"], '
               '"additionalProperties": false}]')


def test_schema_echo_last_attempt_uses_plain_json_object(capsys):
    """qwen3.5-flash can return the schema itself on every strict-schema attempt (CoIN 2026-11-20)."""
    from utils.instance_verifier import verify_candidate_decisions
    formats = []

    def complete(prompt, image, **options):
        formats.append(options['response_format']['type'])
        answer = '{"matches": ["yes"]}' if options['response_format']['type'] == 'json_object' else SCHEMA_ECHO
        options['validator'](answer)
        return answer

    assert verify_candidate_decisions(SimpleNamespace(complete=complete), np.zeros((20, 20, 3), np.uint8),
                                      [[1, 1, 5, 5]], 'picture', {'intrinsic_attributes': 'a picture'}) == ['yes']
    assert formats == ['json_schema', 'json_schema', 'json_object']
    records = [json.loads(l[len('[Instance Verification] '):]) for l in capsys.readouterr().out.splitlines()
               if l.startswith('[Instance Verification] ')]
    assert records[-1]['event'] == 'result' and records[-1]['response_format'] == 'json_object'


def test_exhausted_format_attempts_defer_the_episode_instead_of_ending_the_worker():
    from utils.instance_verifier import verify_candidate_decisions
    from utils.model_client import ModelRequestError

    def complete(prompt, image, **options):
        options['validator'](SCHEMA_ECHO)

    with pytest.raises(ModelRequestError, match='3 format attempts') as error:
        verify_candidate_decisions(SimpleNamespace(complete=complete), np.zeros((20, 20, 3), np.uint8),
                                   [[1, 1, 5, 5]], 'picture', {'intrinsic_attributes': 'a picture'})
    assert error.value.retry_exhausted is True
