import json
import re
from types import SimpleNamespace

import pytest

from utils.llm_filter import LLMFilter
from utils.model_client import ModelClient, ModelOutputTruncatedError, ModelRequestError


@pytest.fixture
def llm_filter(monkeypatch):
    monkeypatch.setenv('PSGNAV_ENV_FILE', '/nonexistent/test.env')
    monkeypatch.setenv('PSGNAV_MODEL_BACKEND', 'openai')
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-key')
    instance = LLMFilter.__new__(LLMFilter)
    instance.model_client = ModelClient()
    instance.verbose = False
    instance.reset_stats()
    return instance


class Candidates:
    def __init__(self, items, kind):
        self.kind = kind
        setattr(self, 'all_combinations' if kind == 'room' else 'all_enumerations', items)
        self.by_room = {}
        self.by_group = {}

    def __len__(self):
        return len(self.items)

    @property
    def items(self):
        return self.all_combinations if self.kind == 'room' else self.all_enumerations

    def _sort(self):
        pass


def candidate(index, probability=.5):
    return dict(room_caption=f'room-{index}', probability=probability, group_idx=index,
                caption_combo=['chair', 'table'], group_node=object(),
                group_nodes=[SimpleNamespace(nodes=[SimpleNamespace(caption='chair')])])


def prompt_items(payload):
    return [(int(local), int(original)) for local, original in re.findall(
        r'^([0-9]+)\. Room: room-([0-9]+)', payload['messages'][0]['content'], re.MULTILINE)]


@pytest.mark.parametrize('kind', ['room', 'group'])
def test_every_uncertain_candidate_is_filtered_and_batch_indices_map_back(llm_filter, monkeypatch, kind):
    items = [candidate(i, 1. if i % 35 == 0 else .5) for i in range(73)]
    candidates = Candidates(items, kind)
    calls = []

    def send(payload):
        rows = prompt_items(payload)
        calls.append(rows)
        assert payload['response_format']['json_schema']['strict'] is True
        return json.dumps({'plausible': [local for local, original in rows if original % 3 != 0]})

    monkeypatch.setattr(llm_filter.model_client, '_send', send)
    method = llm_filter.filter_room_combinations if kind == 'room' else llm_filter.filter_enumerated_groups
    assert method(candidates) is candidates

    expected = [item for i, item in enumerate(items) if i % 35 == 0 or i % 3 != 0]
    assert candidates.items == expected
    assert [len(call) for call in calls] == [32, 32, 6]
    assert [original for call in calls for _, original in call] == [i for i in range(73) if i % 35 != 0]
    assert llm_filter.stats[f'{kind}_kept'] == len(expected)
    assert llm_filter.stats[f'{kind}_filtered'] == 73 - len(expected)
    assert list(candidates.by_room) == [item['room_caption'] for item in expected]
    if kind == 'group':
        assert list(candidates.by_group.values()) == [[item['group_node']] for item in expected]


def test_truncated_reply_splits_without_losing_or_assigning_partial_decisions(llm_filter, monkeypatch):
    sizes = []

    def send(payload):
        rows = prompt_items(payload)
        sizes.append(len(rows))
        if len(rows) > 2:
            raise ModelOutputTruncatedError('finish_reason=length')
        return json.dumps({'plausible': [local for local, original in rows if original in (1, 2, 4)]})

    monkeypatch.setattr(llm_filter.model_client, '_send', send)
    assert llm_filter._filter_batches([candidate(i) for i in range(5)],
                                    llm_filter._build_batch_room_filter_prompt) == {1, 2, 4}
    assert sizes == [5, 2, 3, 1, 2]
    assert len(llm_filter.model_client._cache) == 3


def test_invalid_reply_is_retried_before_caching_and_valid_reply_is_reused(llm_filter, monkeypatch):
    replies = iter(['The answer is 1', '{"plausible":[2]}'])
    calls = []

    def send(payload):
        calls.append(payload)
        return next(replies)

    monkeypatch.setattr(llm_filter.model_client, '_send', send)
    items = [candidate(0), candidate(1)]
    assert llm_filter._filter_batches(items, llm_filter._build_batch_group_filter_prompt) == {1}
    assert len(llm_filter.model_client._cache) == 1
    # Retry prompt has a different key; the validated answer is reusable.
    payload = calls[-1]
    assert llm_filter.model_client.complete(payload['messages'][0]['content'],
        response_format=payload['response_format'],
        validator=lambda text: llm_filter._parse_batch_filter_response(text, 2)) == '{"plausible":[2]}'
    assert len(calls) == 2


def test_later_batch_failure_cannot_partially_filter_candidates(llm_filter, monkeypatch):
    def send(payload):
        if len(prompt_items(payload)) == 32:
            return '{"plausible":[1]}'
        return 'invalid'
    monkeypatch.setattr(llm_filter.model_client, '_send', send)
    items = [candidate(i) for i in range(33)]
    candidates = Candidates(items, 'room')
    with pytest.raises(ModelRequestError):
        llm_filter.filter_room_combinations(candidates)
    assert candidates.items == items
    assert all(value == 0 for value in llm_filter.stats.values())


@pytest.mark.parametrize('failure', ['invalid', 'truncated', 'auth'])
def test_failed_calls_leave_candidates_and_stats_untouched(llm_filter, monkeypatch, failure):
    calls = []

    def send(payload):
        calls.append(payload)
        if failure == 'truncated':
            raise ModelOutputTruncatedError('finish_reason=length')
        if failure == 'auth':
            raise ModelRequestError('Model API HTTP 401')
        return '{"plausible":[true]}'

    monkeypatch.setattr(llm_filter.model_client, '_send', send)
    items = [candidate(0)]
    candidates = Candidates(items, 'room')
    with pytest.raises(ModelRequestError):
        llm_filter.filter_room_combinations(candidates)
    assert candidates.items == items
    assert all(value == 0 for value in llm_filter.stats.values())
    assert not llm_filter.model_client._cache
    assert len(calls) == (1 if failure == 'auth' else 3)


@pytest.mark.parametrize('response,expected', [
    ('{"plausible":[]}', set()),
    ('{"plausible":[1,3]}', {0, 2}),
    ('[{"plausible":[2]}]', {1}),
    ('```json\n{"plausible":[2]}\n```', {1}),
])
def test_complete_decisions_include_empty_and_unambiguous_wrappers(llm_filter, response, expected):
    assert llm_filter._parse_batch_filter_response(response, 3) == expected


@pytest.mark.parametrize('response', [
    '{"plausible":[true]}', '{"plausible":[1.0]}', '{"plausible":["1"]}',
    '{"plausible":[0]}', '{"plausible":[4]}', '{"plausible":[1,1]}',
    '{"plausible":null}', '{"other":[1]}', '{"plausible":[1],"reason":"test"}',
    '[{"plausible":[1]},{"plausible":[2]}]', 'Reason 1: plausible.',
    '{"plausible":[1,',
])
def test_invalid_decisions_are_never_guessed(llm_filter, response):
    with pytest.raises(ValueError):
        llm_filter._parse_batch_filter_response(response, 3)


def test_openrouter_qwen3_vl_schema_omits_unsupported_unique_items(llm_filter, monkeypatch):
    schemas = []

    def send(payload):
        schemas.append(payload['response_format']['json_schema']['schema'])
        return '{"plausible":[1]}'

    monkeypatch.setattr(llm_filter.model_client, '_send', send)
    items = [candidate(0)]
    llm_filter.filter_enumerated_groups(Candidates(items, 'group'))
    assert schemas[0]['properties']['plausible']['uniqueItems'] is True

    llm_filter.model_client.backend = 'openrouter'
    llm_filter.model_client.model = 'qwen/qwen3-vl-8b-instruct'
    llm_filter.filter_enumerated_groups(Candidates(items, 'group'))
    assert 'uniqueItems' not in schemas[1]['properties']['plausible']
    assert schemas[1]['properties']['plausible']['maxItems'] == 1


@pytest.mark.parametrize('kind', ['room', 'group'])
def test_certain_candidates_do_not_call_model(llm_filter, monkeypatch, kind):
    def send(payload):
        pytest.fail('certain candidates must skip the API')
    monkeypatch.setattr(llm_filter.model_client, '_send', send)
    items = [candidate(0, 1.)]
    candidates = Candidates(items, kind)
    method = llm_filter.filter_room_combinations if kind == 'room' else llm_filter.filter_enumerated_groups
    method(candidates)
    assert candidates.items == items
    assert llm_filter.stats[f'{kind}_kept'] == 1
