"""Tests for the M1 re-query analysis. No API calls and no simulator."""
import collections
import json
import os

import pytest

# requery.py turns the API cache off for its own runs (PSGNAV_API_CACHE_SIZE=0 at import).
# Restore the variable so the setting does not leak into other tests in this process.
_cache_size = os.environ.get('PSGNAV_API_CACHE_SIZE')
from outputs.analysis_m1_requery_20261113 import analyze, requery  # noqa: E402
if _cache_size is None:
    os.environ.pop('PSGNAV_API_CACHE_SIZE', None)
else:
    os.environ['PSGNAV_API_CACHE_SIZE'] = _cache_size


def records(*rows):
    """(episode, label, instance, step, answer, original)."""
    return list(rows)


def test_rates_separate_classes():
    data = records(
        (0, 'target', 1, 1, 'yes', 'yes'),
        (0, 'target', 1, 2, 'no', 'no'),
        (0, 'same_category', 2, 1, 'no', 'no'),
        (0, 'same_category', 2, 2, 'no', 'no'),
    )
    row, ratios = analyze.rates(data)
    assert row['target']['boxes'] == 2
    assert row['target']['pooled_yes'] == pytest.approx(.5)
    assert row['same_category']['pooled_no'] == pytest.approx(1)
    assert ratios['no'] == pytest.approx(.5)


def test_macro_rate_weights_tasks_equally():
    # Task 0 answers yes once in ten boxes, task 1 answers yes once in one box.
    data = [(0, 'target', 1, step, 'no', 'no') for step in range(9)]
    data.append((0, 'target', 1, 9, 'yes', 'yes'))
    data.append((1, 'target', 1, 1, 'yes', 'yes'))
    row, _ = analyze.rates(data)
    assert row['target']['pooled_yes'] == pytest.approx(2 / 11)
    assert row['target']['macro_yes'] == pytest.approx((.1 + 1) / 2)


def test_instance_dependence_uses_later_frames_only():
    data = records(
        (0, 'same_category', 5, 3, 'yes', 'yes'),
        (0, 'same_category', 5, 1, 'no', 'no'),   # earlier frame, so it comes first
        (0, 'same_category', 5, 2, 'yes', 'yes'),
    )
    row, _ = analyze.instance_dependence(data)
    # sequence is no, yes, yes: one yes after a non-yes, one yes after a yes
    assert row['distractor_after_nonyes'] == [1, 1]
    assert row['distractor_after_yes'] == [1, 1]


def test_duplicate_boxes_of_one_frame_do_not_count_as_a_second_answer():
    data = records(
        (0, 'same_category', 5, 1, 'yes', 'yes'),
        (0, 'same_category', 5, 1, 'no', 'no'),   # same step, second box of the same frame
    )
    row, _ = analyze.instance_dependence(data)
    assert row['distractor_after_yes'] == [0, 0]
    assert row['distractor_after_nonyes'] == [0, 0]


def test_agreement_ignores_boxes_without_an_original_answer():
    data = records(
        (0, 'target', 1, 1, 'yes', 'yes'),
        (0, 'target', 1, 2, 'no', 'yes'),
        (0, 'target', 1, 3, 'no', None),
    )
    result = analyze.agreement(data)
    assert result['boxes'] == 2
    assert result['identical'] == 1


def test_per_box_aligns_answers_by_box_index(tmp_path):
    sample = [{
        'verification_id': 'abc', 'episode': 0, 'step': 4, 'candidate_count': 3,
        'original_matches': ['no', 'yes', 'unknown'],
        'boxes': [{'box_index': 2, 'answer_index': 2, 'label': 'target', 'gt_instance': 7,
                   'frame_check': 'exact'}],
    }]
    answers = {'abc': {'matches': ['no', 'yes', 'unknown']}}
    out = analyze.per_box(sample, answers)
    assert out == [(0, 'target', 7, 4, 'unknown', 'unknown')]


def test_per_box_skips_candidates_that_were_never_sent():
    # Candidate 0 was never sent for confirmation, so box 1 is the first answer, not the second.
    sample = [{
        'verification_id': 'abc', 'episode': 0, 'step': 4, 'candidate_count': 1,
        'original_matches': ['yes'],
        'boxes': [{'box_index': 1, 'answer_index': 0, 'label': 'target', 'gt_instance': 7,
                   'frame_check': 'close'}],
    }]
    out = analyze.per_box(sample, {'abc': {'matches': ['no']}})
    assert out == [(0, 'target', 7, 4, 'no', 'yes')]


def line(event):
    return '[Instance Verification] ' + json.dumps(event) + '\n'


def test_parse_log_joins_filter_result_to_the_preceding_request():
    request = {'event': 'request', 'stage': 'goal_confirmation', 'verification_id': 'req1',
               'step': 4, 'episode': 0, 'image': {'path': 'images/x.png'}}
    result = {'event': 'result', 'stage': 'goal_confirmation', 'verification_id': 'req1',
              'matches': ['yes']}
    summary = {'event': 'filter_result', 'verification_id': 'sum1', 'step': 4, 'episode': 0}
    requests, results = requery.parse_log([line(request), line(result), line(summary)])
    assert requests['sum1']['verification_id'] == 'req1'
    assert results['req1'] == ['yes']


def test_parse_log_drops_a_filter_result_from_a_different_step():
    request = {'event': 'request', 'stage': 'goal_confirmation', 'verification_id': 'req1',
               'step': 4, 'episode': 0}
    summary = {'event': 'filter_result', 'verification_id': 'sum1', 'step': 9, 'episode': 0}
    requests, _ = requery.parse_log([line(request), line(summary)])
    assert requests == {}


def test_ratio_interval_covers_the_point_estimate():
    data = [(episode, 'target', 1, 1, 'yes', 'yes') for episode in range(20)]
    data += [(episode, 'same_category', 2, 1, 'yes', 'yes') for episode in range(20)]
    data += [(episode, 'same_category', 2, 2, 'no', 'no') for episode in range(20)]
    _row, ratios = analyze.rates(data)
    interval = analyze.ratio_interval(data, 'yes', __import__('random').Random(0))
    assert interval[0] <= ratios['yes'] <= interval[1]


def test_dashscope_payload_carries_the_thinking_switch():
    payload = requery.build_payload('qwen3.5-flash', 'dashscope', 'AAAA', 'prompt', {'type': 'json_schema'}, True)
    assert payload['enable_thinking'] is True
    assert payload['response_format'] == {'type': 'json_schema'}


def test_poe_payload_omits_the_dashscope_extension():
    payload = requery.build_payload('gpt-4o', 'poe', 'AAAA', 'prompt', {'type': 'json_schema'}, False)
    assert 'enable_thinking' not in payload
    assert payload['temperature'] == 0.0
    assert payload['messages'][0]['content'][0]['image_url']['url'] == 'data:image/png;base64,AAAA'


def test_config_names_keep_dashscope_files_and_separate_other_providers():
    assert requery.config_name('qwen3.5-flash', False) == 'qwen3.5-flash_thinkoff'
    assert requery.config_name('qwen3.5-flash', True, 'dashscope') == 'qwen3.5-flash_thinkon'
    assert requery.config_name('gpt-4o', False, 'poe') == 'poe_gpt-4o_thinkoff'


def test_poe_client_requires_its_key(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-dashscope')
    monkeypatch.setenv('POE_API_KEY', '')
    with pytest.raises(ValueError, match='POE_API_KEY'):
        requery.make_client('gpt-4o', 'poe')


def test_poe_client_points_at_poe(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-dashscope')
    monkeypatch.setenv('POE_API_KEY', 'test-poe')
    monkeypatch.delenv('POE_BASE_URL', raising=False)
    client = requery.make_client('gpt-4o', 'poe')
    assert client.url == 'https://api.poe.com/v1/chat/completions'
    assert client.api_key == 'test-poe'
    assert client.model == 'gpt-4o'


def test_siliconflow_client_points_at_siliconflow(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-dashscope')
    monkeypatch.setenv('SILICONFLOW_API_KEY', 'test-sf')
    monkeypatch.delenv('SILICONFLOW_BASE_URL', raising=False)
    client = requery.make_client('Qwen/Qwen3-VL-8B-Instruct', 'siliconflow')
    assert client.url == 'https://api.siliconflow.cn/v1/chat/completions'
    assert client.api_key == 'test-sf'
    assert client.model == 'Qwen/Qwen3-VL-8B-Instruct'


def test_siliconflow_payload_omits_the_dashscope_extension():
    payload = requery.build_payload('Qwen/Qwen3-VL-8B-Instruct', 'siliconflow', 'AAAA', 'prompt',
                                    {'type': 'json_schema'}, False)
    assert 'enable_thinking' not in payload


def test_hub_model_names_become_flat_file_names():
    assert requery.config_name('Qwen/Qwen3-VL-8B-Instruct', False, 'siliconflow') == \
        'siliconflow_Qwen_Qwen3-VL-8B-Instruct_thinkoff'


def test_macro_ratio_weights_tasks_equally():
    # Task 0: target yes 1/1, same-category yes 1/10. Task 1: target yes 0/1, same-category yes 1/2.
    task0 = [(0, 'target', 1, 1, 'yes', None)] + [
        (0, 'same_category', 2, s, 'yes' if s == 0 else 'no', None) for s in range(10)]
    task1 = [(1, 'target', 1, 1, 'no', None), (1, 'same_category', 2, 1, 'yes', None),
             (1, 'same_category', 2, 2, 'no', None)]
    assert analyze.group_ratio([task0, task1], 'yes', 'macro') == pytest.approx(.5 / ((.1 + .5) / 2))
    assert analyze.group_ratio([task0, task1], 'yes', 'pooled') == pytest.approx(.5 / (2 / 12))


def test_shared_errors_counts_boxes_and_instances():
    first = {('a', 0): (0, 'same_category', 5, 'yes'), ('b', 0): (0, 'same_category', 5, 'no'),
             ('c', 0): (0, 'same_category', 6, 'no'), ('d', 0): (0, 'target', 1, 'yes')}
    second = {('a', 0): (0, 'same_category', 5, 'yes'), ('b', 0): (0, 'same_category', 5, 'no'),
              ('c', 0): (0, 'same_category', 6, 'yes'), ('d', 0): (0, 'target', 1, 'yes')}
    result = analyze.shared_errors(first, second)
    assert result['boxes'] == {'after_yes': [1, 1], 'after_nonyes': [1, 2]}
    # instance 5: first said yes somewhere, second too; instance 6: first never, second yes
    assert result['instances'] == {'after_yes': [1, 1], 'after_nonyes': [1, 1]}


def test_recorded_answers_reuse_the_logged_matches():
    sample = [{'verification_id': 'x', 'original_matches': ['yes']},
              {'verification_id': 'y', 'original_matches': None}]
    assert analyze.recorded_answers(sample) == {'x': {'matches': ['yes']}}
