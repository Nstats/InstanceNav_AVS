"""M7 component ablations: T0 (release for one approach), A0 (no approach) and approach logging."""
import json
import os
import subprocess

import numpy as np
import pytest

from configs.navigation_config import validate_instance_ablation
from test_instance_ablation import ablation  # noqa: F401  (fixture)
from test_instance_confirmation_controls import controls, observe_scenegraph  # noqa: F401
from test_instance_constraints import navigation, post_navigation, observe_navigation  # noqa: F401
from tools.run_instance_ablation import settings


def records(capsys, event):
    out = capsys.readouterr().out
    rows = [json.loads(line.split(' ', 2)[2]) for line in out.splitlines()
            if line.startswith('[Instance Verification] ')]
    return [row for row in rows if row.get('event') == event]


def confirmation_calls(state):
    return sum(stage == 'goal_confirmation' for stage, _ in state['stage_calls'])


def at(fixture, step, position, answer, path='navigation'):
    agent, _, _, state = fixture
    agent.last_gps = np.asarray(position, dtype=float)
    state['stage_answers']['goal_confirmation'] = [answer]
    (observe_navigation if path == 'navigation' else observe_scenegraph)(fixture, step)


@pytest.mark.parametrize('path', ['navigation', 'scenegraph'])
def test_t0_releases_for_one_approach_then_the_candidate_returns(ablation, monkeypatch, capsys, path):
    agent, manager, _, state = ablation
    monkeypatch.setenv('INSTANCE_REJECTION_POLICY', 'non_yes')
    monkeypatch.setenv('INSTANCE_REJECTION_SCOPE', 'approach')
    at(ablation, 1, [0, 0], 'unknown', path)
    assert not agent.found_goal and not agent.found_possible_goal
    assert not agent.instance_rejections  # Released, never excluded.
    assert not any(track.vetoed for track in manager._positive_evidence.tracks)
    assert len(records(capsys, 'approach_release')) == 1
    # Still in view: the same approach, so it is neither asked nor approached.
    at(ablation, 3, [0, 0], 'yes', path)
    assert confirmation_calls(state) == 1 and not agent.found_goal and not agent.found_possible_goal
    # Out of view long enough, but the agent has not moved 1 m: still the same approach.
    at(ablation, 9, [.5, 0], 'yes', path)
    assert confirmation_calls(state) == 1 and not agent.found_goal
    # Out of view again for >= 5 steps and moved >= 1 m: a new approach, asked again.
    at(ablation, 15, [1.5, 0], 'yes', path)
    assert confirmation_calls(state) == 2 and agent.found_goal
    ended = records(capsys, 'approach_release_end')
    assert len(ended) == 1 and ended[0]['answer'] == 'unknown' and ended[0]['moved_m'] >= 1


def test_t0_no_releases_only_on_explicit_no(ablation, monkeypatch):
    agent, _, _, state = ablation
    monkeypatch.setenv('INSTANCE_REJECTION_POLICY', 'no')
    monkeypatch.setenv('INSTANCE_REJECTION_SCOPE', 'approach')
    at(ablation, 1, [0, 0], 'unknown')
    at(ablation, 2, [0, 0], 'no')
    assert confirmation_calls(state) == 2
    at(ablation, 3, [0, 0], 'yes')
    assert confirmation_calls(state) == 2 and not agent.found_goal and not agent.instance_rejections


def test_t0_release_clears_the_current_approach_target(ablation, monkeypatch, capsys):
    agent, manager, _, state = ablation
    monkeypatch.setenv('INSTANCE_REJECTION_POLICY', 'non_yes')
    monkeypatch.setenv('INSTANCE_REJECTION_SCOPE', 'approach')
    # One fixed object; only its distance from the agent changes.
    agent.get_goal_gps = lambda obs, angle, distance: np.array([6., 0.])
    agent.depth[:] = 6.  # Far: approached, not yet verified.
    at(ablation, 1, [0, 0], 'yes')
    assert agent.found_possible_goal and confirmation_calls(state) == 0
    agent.depth[:] = 3.
    at(ablation, 2, [0, 0], 'no')
    assert not agent.found_possible_goal and manager._post_possible_track_id is None
    ends = records(capsys, 'approach_end')
    assert [row['reason'] for row in ends] == ['approach_release']


def test_episode_scope_keeps_e3_permanent(ablation, monkeypatch):
    agent, manager, _, state = ablation
    monkeypatch.setenv('INSTANCE_REJECTION_POLICY', 'no')
    at(ablation, 1, [0, 0], 'no')
    at(ablation, 20, [5, 0], 'yes')
    assert len(agent.instance_rejections) == 1 and not agent.found_goal and confirmation_calls(state) == 1


@pytest.mark.parametrize('path', ['navigation', 'scenegraph'])
def test_a0_never_approaches_but_still_verifies_and_locks(ablation, monkeypatch, path):
    agent, _, _, state = ablation
    monkeypatch.setenv('ENABLE_CANDIDATE_APPROACH', 'false')
    agent.depth[:] = 6.
    at(ablation, 1, [0, 0], 'yes', path)
    assert not agent.found_possible_goal and not agent.found_goal and confirmation_calls(state) == 0
    agent.depth[:] = 3.
    at(ablation, 2, [0, 0], 'unknown', path)
    assert not agent.found_possible_goal and confirmation_calls(state) == 1
    at(ablation, 3, [0, 0], 'yes', path)
    assert agent.found_goal


def test_b_approaches_far_and_unconfirmed_near_candidates(ablation):
    agent, _, _, _ = ablation
    agent.depth[:] = 6.
    at(ablation, 1, [0, 0], 'yes')
    assert agent.found_possible_goal and not agent.found_goal
    agent.depth[:] = 3.
    at(ablation, 2, [0, 0], 'unknown')
    assert agent.found_possible_goal and not agent.found_goal


def test_approach_log_records_start_reach_and_reselection(ablation, capsys):
    agent, manager, _, _ = ablation
    agent.get_goal_gps = lambda obs, angle, distance: np.array([6., 0.])
    agent.depth[:] = 6.  # Approach targets come from unverified (far) candidates.
    at(ablation, 1, [0, 0], 'unknown')
    starts = records(capsys, 'approach_start')
    assert len(starts) == 1 and starts[0]['reselected_after_reach'] is False
    assert starts[0]['within_verification_range'] is False
    at(ablation, 2, [0, 0], 'unknown')
    assert not records(capsys, 'approach_start')  # Same target: no new record.
    manager.log_approach_end('reached')
    agent.found_possible_goal = False
    assert records(capsys, 'approach_end')[0]['reason'] == 'reached'
    at(ablation, 3, [0, 0], 'unknown')
    again = records(capsys, 'approach_start')
    assert len(again) == 1 and again[0]['reselected_after_reach'] is True


def test_b_near_non_yes_keeps_the_existing_target(ablation, capsys):
    # A near non-yes candidate is not kept for that frame, so it does not replace or
    # clear the current approach target; the agent keeps heading to it (B behaviour).
    agent, manager, _, _ = ablation
    agent.get_goal_gps = lambda obs, angle, distance: np.array([6., 0.])
    agent.depth[:] = 6.
    at(ablation, 1, [0, 0], 'unknown')
    target = agent.possible_goal_temp_gps.copy()
    agent.depth[:] = 3.
    at(ablation, 2, [0, 0], 'unknown')
    assert agent.found_possible_goal and np.allclose(agent.possible_goal_temp_gps, target)
    assert not records(capsys, 'approach_end')


def test_approach_log_does_not_change_b_decisions(ablation, monkeypatch):
    agent, manager, _, state = ablation
    at(ablation, 1, [0, 0], 'unknown')
    before = (agent.found_possible_goal, manager._post_possible_track_id, confirmation_calls(state))
    monkeypatch.setattr(manager, '_log_approach_target', lambda *a, **k: None)
    at(ablation, 2, [0, 0], 'unknown')
    assert (agent.found_possible_goal, manager._post_possible_track_id) == before[:2]
    assert confirmation_calls(state) == before[2] + 1


@pytest.mark.parametrize('override', [
    {'INSTANCE_REJECTION_SCOPE': 'typo'},
    {'INSTANCE_REJECTION_SCOPE': 'approach'},  # Requires a rejection policy.
    {'ENABLE_CANDIDATE_APPROACH': 'false', 'ENABLE_INSTANCE_CONFIRMATION': 'false'},
    {'ENABLE_CANDIDATE_APPROACH': 'maybe'},
])
def test_invalid_m7_combinations_fail_before_navigation(monkeypatch, override):
    for key, value in {**settings('B'), **override}.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        validate_instance_ablation()


@pytest.mark.parametrize('experiment', ['B', 'T0', 'T0no', 'A0', 'RA'])
def test_m7_presets_validate(monkeypatch, experiment):
    for key, value in settings(experiment).items():
        monkeypatch.setenv(key, value)
    result = validate_instance_ablation()
    assert result['rejection_scope'] == ('approach' if experiment.startswith('T0') else 'episode')
    assert result['candidate_approach'] == (experiment != 'A0')


def test_shell_launcher_exports_m7_switches(tmp_path):
    fake_python = tmp_path / 'python'
    capture = tmp_path / 'capture'
    fake_python.write_text(
        '#!/bin/bash\nprintf "%s %s\\n" "$INSTANCE_REJECTION_SCOPE" "$ENABLE_CANDIDATE_APPROACH" '
        '>> "$SWITCH_TEST_CAPTURE"\nif [[ "$1" == "-" ]]; then cat >/dev/null; fi\n')
    fake_python.chmod(0o755)
    env = {**os.environ, 'PSGNAV_PYTHON': str(fake_python), 'SWITCH_TEST_CAPTURE': str(capture)}
    for name in ('INSTANCE_REJECTION_SCOPE', 'ENABLE_CANDIDATE_APPROACH', 'ENABLE_CONTRADICTION_CHECK'):
        env.pop(name, None)
    base = ['bash', 'scripts/run_instance100.sh', '--check-only']
    result = subprocess.run(base, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines()[-1] == 'episode true'
    result = subprocess.run([*base, '--instance-rejection-scope', 'APPROACH',
                             '--instance-rejection-policy', 'non_yes',
                             '--enable-candidate-approach', 'NO'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines()[-1] == 'approach false'
    assert 'INSTANCE_REJECTION_SCOPE: approach' in result.stdout
    assert 'ENABLE_CANDIDATE_APPROACH: false' in result.stdout
    count = len(capture.read_text().splitlines())
    for options in (['--instance-rejection-scope', 'bad'], ['--enable-candidate-approach', 'bad']):
        result = subprocess.run([*base, *options], env=env, capture_output=True, text=True)
        assert result.returncode == 2, options
    assert len(capture.read_text().splitlines()) == count  # Invalid flags never start Python.
