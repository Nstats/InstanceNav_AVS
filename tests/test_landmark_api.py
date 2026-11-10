import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from utils.instance_verifier import verify_candidates
from utils.landmark_selector import LandmarkSelector
from utils.model_client import ModelRequestError


def test_pairwise_batch_preserves_all_votes_and_order():
    selector=LandmarkSelector.__new__(LandmarkSelector)
    selector.agent=SimpleNamespace(goal_attributes=None)
    selector._sample_scenegraph=lambda _: {'id':'s'}
    selector._add_scenegraph_semantics_to_landmark=lambda landmark,sg:landmark
    selector._create_pairwise_comparison_prompt=lambda a,b,goal,i,j:f'{i}:{j}'
    captured=[]
    selector.get_llm_responses=lambda prompts:captured.extend(prompts) or ['A']*len(prompts)
    landmarks=[{'win_count':0,'total_comparisons':0} for _ in range(3)]
    result=selector._compare_landmarks_pairwise(landmarks,None,'chair')
    assert len(captured)==9
    assert result['total_comparisons']==9
    assert result['win_counts']=={0:6,1:3,2:0}
    assert [lm['total_comparisons'] for lm in landmarks]==[6,6,6]


def test_invalid_comparison_does_not_silently_favor_first_landmark():
    selector=LandmarkSelector.__new__(LandmarkSelector)
    assert selector._parse_comparison_winner('B',2,4)==4
    assert selector._parse_comparison_winner('A: near the kitchen',2,4)==2
    with pytest.raises(ModelRequestError):selector._parse_comparison_winner('API error',2,4)


@pytest.mark.parametrize('enabled', [None, 'true', 'false'])
def test_exploration_description_switch_preserves_instance_verification(monkeypatch, enabled):
    switch = 'ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION'
    if enabled is None:
        monkeypatch.delenv(switch, raising=False)
    else:
        monkeypatch.setenv(switch, enabled)
    monkeypatch.setenv('LANDMARK_COMPARISON_REPEATS', '1')
    monkeypatch.setenv('LANDMARK_COMPARISON_WORKERS', '1')
    attributes = {'intrinsic_attributes': 'red striped upholstery',
                  'extrinsic_attributes': 'beside the piano'}
    search_prompts, verification_prompts = [], []

    def complete(prompt, image=None, **options):
        if image is None:
            search_prompts.append(prompt)
            return 'A'
        verification_prompts.append(prompt)
        return '{"matches":["yes"]}'

    selector = LandmarkSelector.__new__(LandmarkSelector)
    selector.agent = SimpleNamespace(goal_attributes=attributes.copy())
    selector.model_client = SimpleNamespace(complete=complete, concurrency=1)
    selector._get_exploration_context = lambda _: 'Living room partially explored.'
    landmarks = [
        {'semantics': 'open area', 'win_count': 0, 'total_comparisons': 0,
         'geometry': {'room_name': 'living room', 'room_probability': 0.8,
                      'distance_to_agent_meters': distance, 'exploration_pct': 0.2,
                      'near_frontier': True,
                      'nearby_objects': [{'caption': 'table', 'distance_meters': 1.0}]}}
        for distance in (2.0, 4.0)
    ]
    result = selector._compare_landmarks_pairwise(landmarks, None, 'chair')
    assert result['total_comparisons'] == len(search_prompts) == 1
    prompt = search_prompts[0]
    assert 'searching for "chair"' in prompt
    assert 'Living room partially explored.' in prompt
    assert 'near table (1.0m)' in prompt
    for description in attributes.values():
        assert (description in prompt) == (enabled != 'false')
    assert selector.agent.goal_attributes == attributes

    kept = verify_candidates(selector.model_client, np.zeros((20, 20, 3), dtype=np.uint8),
                             [[1, 1, 10, 10]], 'chair', selector.agent.goal_attributes)
    assert kept == [0] and len(verification_prompts) == 1
    assert all(description in verification_prompts[0] for description in attributes.values())


@pytest.mark.parametrize('environment_value,cli_value,expected', [
    (None, None, 'false'),
    ('NO', None, 'false'),
    ('true', 'NO', 'false'),
    ('false', 'YES', 'true'),
])
def test_launcher_exports_exploration_description_switch(tmp_path, environment_value, cli_value, expected):
    switch = 'ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION'
    flag = '--enable-instance-description-for-exploration'
    capture = tmp_path / 'capture'
    fake_python = tmp_path / 'python'
    fake_python.write_text(
        '#!/bin/bash\nif [[ "$1" == "-" ]]; then cat >/dev/null; fi\n'
        'printf "%s %s\\n" "$ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION" '
        '"$ENABLE_INSTANCE_CONFIRMATION" >> "$SWITCH_TEST_CAPTURE"\n')
    fake_python.chmod(0o755)
    env = {**os.environ, 'PSGNAV_PYTHON': str(fake_python), 'SWITCH_TEST_CAPTURE': str(capture),
           'ENABLE_INSTANCE_CONFIRMATION': 'true', 'INSTANCE_CONFIRMATION_MODE': 'positive_match',
           'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK': 'false',
           'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY': 'false', 'INSTANCE_COLLECT_ONLY': 'false'}
    env.pop('ENABLE_CONTRADICTION_CHECK', None)
    env.pop(switch, None)
    if environment_value is not None:
        env[switch] = environment_value
    command = ['bash', 'scripts/run_instance100.sh', '--check-only']
    if cli_value is not None:
        command.extend([flag, cli_value])
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == [f'{expected} true'] * 2
    assert f'{switch}: {expected}' in result.stdout

    capture.unlink()
    for options in ([flag, 'bad'], [flag], [flag, '--check-only']):
        result = subprocess.run(['bash', 'scripts/run_instance100.sh', *options], env=env,
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 2, options
        assert not capture.exists()  # Reject invalid input before loading Python or models.
