import base64
import os
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import hashlib
import json
import threading
import time

# The endpoint fixture replaces time.sleep for retry backoff. Keep the real sleeper.
_REAL_SLEEP = time.sleep

from PIL import Image
import pytest

from utils.model_client import ModelClient, ModelOutputTruncatedError, ModelRequestError


@pytest.fixture
def endpoint(monkeypatch):
    state = {'calls': [], 'responses': [], 'headers': []}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['calls'].append(payload)
            state['headers'].append({
                'authorization': self.headers.get('Authorization'),
                'referer': self.headers.get('HTTP-Referer'),
                'title': self.headers.get('X-OpenRouter-Title'),
            })
            status, body = state['responses'].pop(0) if state['responses'] else (
                200, {'choices':[{'finish_reason':'stop','message':{'content':'yes'}}]})
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())
    server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
    monkeypatch.setenv('no_proxy', '127.0.0.1,localhost')
    monkeypatch.setenv('PSGNAV_ENV_FILE','/nonexistent/test.env')
    monkeypatch.setenv('PSGNAV_MODEL_BACKEND','openai')
    monkeypatch.setenv('DASHSCOPE_API_KEY','test-key')
    monkeypatch.setenv('DASHSCOPE_BASE_URL',f'http://127.0.0.1:{server.server_port}/v1')
    monkeypatch.setenv('QWEN_MODEL_NAME','qwen3-vl-flash')
    monkeypatch.setenv('PSGNAV_API_RETRIES','1')
    monkeypatch.setattr('utils.model_client.time.sleep',lambda delay: None)
    yield state
    server.shutdown()
    server.server_close()
    worker.join()


def test_openrouter_uses_its_own_key_and_omits_dashscope_switch(endpoint, monkeypatch):
    monkeypatch.setenv('PSGNAV_MODEL_BACKEND', 'openrouter')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'or-test-key')
    monkeypatch.setenv('QWEN_MODEL_NAME', 'google/gemini-2.5-flash')
    monkeypatch.setenv('OPENROUTER_BASE_URL', os.environ['DASHSCOPE_BASE_URL'])
    monkeypatch.setenv('OPENROUTER_HTTP_REFERER', 'https://localhost')
    monkeypatch.setenv('OPENROUTER_TITLE', 'psg-nav-instance')
    client = ModelClient()
    assert client.complete('image question', Image.new('RGB', (10, 10), 'red')) == 'yes'
    payload = endpoint['calls'][0]
    assert payload['model'] == 'google/gemini-2.5-flash'
    assert 'enable_thinking' not in payload
    assert payload['temperature'] == 0.0
    url = payload['messages'][0]['content'][0]['image_url']['url']
    assert url.startswith('data:image/png;base64,')
    headers = endpoint['headers'][0]
    assert headers['authorization'] == 'Bearer or-test-key'
    assert headers['referer'] == 'https://localhost'
    assert headers['title'] == 'psg-nav-instance'


def test_cache_metadata_is_available_inside_validator_and_is_thread_local(endpoint):
    client = ModelClient()
    seen = []
    def ask(prompt):
        return client.complete(prompt, validator=lambda answer: seen.append(client.last_request_metadata()))
    ask('repeat')
    ask('repeat')
    assert [r['cache_hit'] for r in seen] == [False, True]
    assert seen[0]['cache_key'] == seen[1]['cache_key']
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(ask, ['other', 'repeat']))
    assert sorted(r['cache_hit'] for r in seen[2:]) == [False, True]
    assert client.last_request_metadata() == seen[1]
    assert set(seen[0]) == {'cache_hit', 'cache_key'}


def test_siliconflow_uses_its_own_key_and_omits_dashscope_switch(endpoint, monkeypatch):
    monkeypatch.setenv('PSGNAV_MODEL_BACKEND', 'siliconflow')
    monkeypatch.setenv('SILICONFLOW_API_KEY', 'sf-test-key')
    monkeypatch.setenv('QWEN_MODEL_NAME', 'Qwen/Qwen3-VL-8B-Instruct')
    monkeypatch.setenv('SILICONFLOW_BASE_URL', os.environ['DASHSCOPE_BASE_URL'])
    client = ModelClient()
    assert client.complete('text question') == 'yes'
    payload = endpoint['calls'][0]
    assert payload['model'] == 'Qwen/Qwen3-VL-8B-Instruct'
    assert 'enable_thinking' not in payload
    assert endpoint['headers'][0]['authorization'] == 'Bearer sf-test-key'
    assert endpoint['headers'][0]['referer'] is None


def test_openrouter_requires_its_own_key(monkeypatch):
    monkeypatch.setenv('PSGNAV_ENV_FILE', '/nonexistent/test.env')
    monkeypatch.setenv('PSGNAV_MODEL_BACKEND', 'openrouter')
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    with pytest.raises(ValueError, match='OPENROUTER_API_KEY'):
        ModelClient()


def test_multimodal_rgb_and_text_share_api(endpoint):
    client=ModelClient()
    assert client.complete('text question') == 'yes'
    assert client.complete('image question', Image.new('RGB',(10,10),'red')) == 'yes'
    payload=endpoint['calls'][1]
    assert payload['model']=='qwen3-vl-flash'
    assert payload['enable_thinking'] is False
    url=payload['messages'][0]['content'][0]['image_url']['url']
    decoded=Image.open(BytesIO(base64.b64decode(url.split(',',1)[1])))
    assert decoded.getpixel((0,0)) == (255,0,0)


def test_same_pixels_reuse_png_encoding_without_reusing_different_answers(endpoint, monkeypatch):
    client = ModelClient()
    saves = []
    original = Image.Image.save
    def save(image, *args, **kwargs):
        saves.append(image.size)
        return original(image, *args, **kwargs)
    monkeypatch.setattr(Image.Image, 'save', save)
    client.complete('first', Image.new('RGB', (10, 10), 'red'))
    client.complete('second', Image.new('RGB', (10, 10), 'red'))
    assert len(saves) == 1 and len(endpoint['calls']) == 2
    urls = [call['messages'][0]['content'][0]['image_url']['url'] for call in endpoint['calls']]
    assert urls[0] == urls[1]
    client.complete('second', Image.new('RGB', (10, 10), 'blue'))
    assert len(saves) == 2 and len(endpoint['calls']) == 3


def test_identical_parallel_requests_sent_once_and_changed_image_not_cached(endpoint):
    client=ModelClient()
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(client.complete,['same']*4))==['yes']*4
    assert len(endpoint['calls'])==1
    client.complete('same',Image.new('RGB',(10,10),'red'))
    client.complete('same',Image.new('RGB',(10,10),'blue'))
    assert len(endpoint['calls'])==3


def test_instance_verification_logs_cached_output_without_an_extra_request(endpoint, monkeypatch, capsys):
    import numpy as np
    from utils.instance_verifier import verify_candidates

    monkeypatch.setenv('PSGNAV_API_CACHE_SIZE', '8')
    answer = '{"matches":["unknown","yes"]}'
    endpoint['responses'] = [(200, {'choices': [{'finish_reason': 'stop', 'message': {'content': answer}}]})]
    client = ModelClient()
    for step in (30, 31):
        assert verify_candidates(client, np.zeros((20, 20, 3), dtype=np.uint8),
                                 [[1, 1, 5, 5], [8, 8, 12, 12]], 'chair',
                                 {'intrinsic_attributes': 'A beige chair.'},
                                 log_context={'step': step}) == [1]

    assert len(endpoint['calls']) == 1 and client.stats['cache_hits'] == 1
    prefix = '[Instance Verification] '
    records = [json.loads(line[len(prefix):]) for line in capsys.readouterr().out.splitlines()
               if line.startswith(prefix)]
    assert [r['event'] for r in records] == ['raw_response', 'result'] * 2
    assert [r['step'] for r in records] == [30, 30, 31, 31]
    assert all(r['answer'] == answer for r in records if r['event'] == 'raw_response')
    assert len({r['verification_id'] for r in records}) == 2
    assert 'test-key' not in json.dumps(records) and 'data:image' not in json.dumps(records)


def test_http_400_records_the_rejected_call_without_secrets_or_image_bytes(endpoint, monkeypatch, capsys):
    monkeypatch.setenv('PSGNAV_MODEL_BACKEND', 'siliconflow')
    monkeypatch.setenv('SILICONFLOW_API_KEY', 'sf-test-key')
    monkeypatch.setenv('QWEN_MODEL_NAME', 'Qwen/Qwen3-VL-8B-Instruct')
    monkeypatch.setenv('SILICONFLOW_BASE_URL', os.environ['DASHSCOPE_BASE_URL'])
    endpoint['responses'] = [(400, {
        'code': 20015,
        'message': 'The parameter is invalid. Please check again. key=sf-test-key',
        'data': None,
    })]
    client = ModelClient()
    image = Image.new('RGB', (6, 4), 'red')
    with pytest.raises(ModelRequestError, match='HTTP 400: 20015 The parameter is invalid') as exc:
        client.complete('describe this plant', image,
                        response_format={'type': 'json_schema', 'json_schema': {'name': 'instance_matches'}},
                        max_tokens=128)
    assert 'sf-test-key' not in str(exc.value)
    assert len(endpoint['calls']) == 1
    line = next(item for item in capsys.readouterr().out.splitlines() if item.startswith('[API HTTP 400] '))
    record = json.loads(line.split('] ', 1)[1])
    assert record['backend'] == 'siliconflow'
    assert record['model'] == 'Qwen/Qwen3-VL-8B-Instruct'
    assert record['attempt'] == 1
    assert record['request']['max_tokens'] == 128
    assert record['request']['response_format']['json_schema']['name'] == 'instance_matches'
    content = record['request']['messages'][0]['content']
    assert content[1] == {'type': 'text', 'text': 'describe this plant'}
    assert content[0]['image']['width'] == 6
    assert content[0]['image']['height'] == 4
    assert content[0]['image']['format'] == 'PNG'
    assert len(content[0]['image']['sha256']) == 64
    assert record['response']['status'] == 400
    assert record['response']['body']['code'] == 20015
    assert record['response']['body']['message'] == 'The parameter is invalid. Please check again. key=[redacted]'
    assert 'sf-test-key' not in line
    assert 'data:image' not in line
    assert client.complete('describe this plant', image,
                           response_format={'type': 'json_schema', 'json_schema': {'name': 'instance_matches'}},
                           max_tokens=128) == 'yes'
    assert len(endpoint['calls']) == 2


def test_json_decode_error_is_retried_then_accepted(endpoint):
    endpoint['responses'] = [(400, {'code': 20015, 'message': 'JSON decode error', 'data': None})]
    client = ModelClient()
    assert client.complete('question') == 'yes'
    assert len(endpoint['calls']) == 2


def test_json_decode_error_exhaustion_is_marked(endpoint):
    rejected = (400, {'code': 20015, 'message': 'JSON decode error', 'data': None})
    endpoint['responses'] = [rejected, rejected]
    client = ModelClient()
    with pytest.raises(ModelRequestError, match='JSON decode error after 2 attempts') as exc:
        client.complete('question')
    assert exc.value.retry_exhausted is True
    assert len(endpoint['calls']) == 2


def test_rate_limit_retries_but_auth_error_is_not_cached(endpoint):
    endpoint['responses']=[(429,{'error':'limited'})]
    client=ModelClient()
    assert client.complete('retry')=='yes'
    assert len(endpoint['calls'])==2
    endpoint['responses']=[(401,{'error':'secret must not appear'})]
    with pytest.raises(ModelRequestError, match='HTTP 401') as exc:
        client.complete('auth')
    assert 'secret' not in str(exc.value)
    assert client.complete('auth')=='yes'
    assert len(endpoint['calls'])==4


@pytest.mark.parametrize('body',[
    {'choices':[]},
    {'choices':[{'finish_reason':'length','message':{'content':'partial'}}]},
    {'choices':[{'finish_reason':'stop','message':{'content':''}}]},
])
def test_invalid_answers_fail_instead_of_becoming_navigation_evidence(endpoint,body):
    endpoint['responses']=[(200,body)]
    with pytest.raises(ModelRequestError):
        ModelClient().complete('question')


def test_retry_budget_is_bounded(endpoint):
    endpoint['responses']=[(503,{}),(503,{}),(503,{})]
    with pytest.raises(ModelRequestError,match='after 2 attempts'):
        ModelClient().complete('question')
    assert len(endpoint['calls'])==2


@pytest.mark.parametrize('reason,expected', [
    ('length', ModelOutputTruncatedError), ('content_filter', ModelRequestError),
])
def test_only_length_failures_can_be_recovered_by_splitting(endpoint, reason, expected):
    endpoint['responses'] = [(200, {'choices': [
        {'finish_reason': reason, 'message': {'content': '{"plausible":[1'}}
    ], 'usage': {'completion_tokens': 4096}})]
    client = ModelClient()
    with pytest.raises(expected) as exc:
        client.complete('filter candidates')
    assert type(exc.value) is expected
    assert not client._cache
    assert len(endpoint['calls']) == 1


def test_invalid_structured_reply_is_not_cached_and_valid_reply_is_reused(endpoint):
    from utils.instance_verifier import parse_matches
    def response(text):
        return (200, {'choices':[{'finish_reason':'stop','message':{'content':text}}]})
    endpoint['responses'] = [response('{"matches":["yes"]}'),
                             response('{"matches":["unknown","yes"]}')]
    client = ModelClient()
    format_spec = {'type':'json_schema','json_schema':{'name':'test','schema':{'type':'object'}}}
    options = {'response_format':format_spec, 'validator':lambda value:parse_matches(value,2)}
    with pytest.raises(ValueError, match='wrong matches type or length'):
        client.complete('two candidates', **options)
    assert client.complete('two candidates', **options) == '{"matches":["unknown","yes"]}'
    assert client.complete('two candidates', **options) == '{"matches":["unknown","yes"]}'
    assert len(endpoint['calls']) == 2
    assert all(call['response_format'] == format_spec for call in endpoint['calls'])
    # Ordinary text calls retain their original payload and a separate cache key.
    assert client.complete('two candidates') == 'yes'
    assert 'response_format' not in endpoint['calls'][-1]


def test_multi_image_payload_preserves_all_views_order_and_cache_identity(endpoint):
    client = ModelClient()
    frames = [Image.new('RGB', (8, 8), color) for color in ('red', 'blue', 'green')]
    client.complete('all observations', images=frames, max_tokens=512)
    payload = endpoint['calls'][0]
    content = payload['messages'][0]['content']
    urls = [part['image_url']['url'] for part in content if part['type'] == 'image_url']
    pixels = [Image.open(BytesIO(base64.b64decode(url.split(',', 1)[1]))).getpixel((0, 0))
              for url in urls]
    assert pixels == [(255, 0, 0), (0, 0, 255), (0, 128, 0)]
    assert [part['text'] for part in content if part['type'] == 'text'] == [
        'Image 0:', 'Image 1:', 'Image 2:', 'all observations']
    assert payload['max_tokens'] == 512
    client.complete('all observations', images=frames, max_tokens=512)
    assert len(endpoint['calls']) == 1
    client.complete('all observations', images=list(reversed(frames)), max_tokens=512)
    client.complete('all observations', images=frames[:2], max_tokens=512)
    assert len(endpoint['calls']) == 3
    assert client.stats['vision_seconds'] > 0 and client.stats['text_seconds'] == 0
    with pytest.raises(ValueError, match='not both'):
        client.complete('ambiguous', frames[0], images=frames)


@pytest.mark.parametrize('success_attempt', [1, 2, 3, None])
def test_post_check_owns_three_total_http_and_format_attempts(endpoint, monkeypatch, capsys, success_attempt):
    from utils.instance_verifier import verify_post_confirmation

    def response(answer):
        return 200, {'choices': [{'finish_reason': 'stop', 'message': {'content': answer}}]}

    monkeypatch.setenv('PSGNAV_API_RETRIES', '9')  # Must not multiply the outer budget.
    monkeypatch.delenv('PSGNAV_LOG_FILE', raising=False)
    failures = [
        (503, {'error': {'code': 'overloaded', 'message': 'Provider unavailable'}}),
        response('{"decision":"no_conflict"}'),
        (401, {'error': {'code': 'InvalidKey', 'message': 'Rejected test-key'}}),
    ]
    valid = response('{"decision":"unknown","image_index":-1,"reason":""}')
    endpoint['responses'] = (failures[:success_attempt - 1] + [valid]
                             if success_attempt is not None else failures)
    client = ModelClient()
    args = client, [Image.new('RGB', (8, 8), 'red')], 'chair', {'intrinsic_attributes': 'A beige chair.'}
    if success_attempt is not None:
        assert verify_post_confirmation(*args)['decision'] == 'unknown'
        assert len(endpoint['calls']) == success_attempt
    else:
        with pytest.raises(ModelRequestError, match='failed after 3 attempts') as exc:
            verify_post_confirmation(*args)
        assert len(endpoint['calls']) == 3 and not client._cache
        assert 'Provider unavailable' in str(exc.value) and 'expected exactly' in str(exc.value)
        assert 'InvalidKey' in str(exc.value) and 'test-key' not in str(exc.value)
        assert 'attempt 1:' in str(exc.value) and 'attempt 3:' in str(exc.value)
        assert str(exc.value) in capsys.readouterr().err
    assert all(call['response_format']['json_schema']['strict'] for call in endpoint['calls'])
    assert all(call['max_tokens'] == 512 for call in endpoint['calls'])


def test_post_check_retries_truncated_output_and_stops_on_a_valid_conflict(endpoint, monkeypatch):
    from utils.instance_verifier import verify_post_confirmation
    monkeypatch.delenv('PSGNAV_LOG_FILE', raising=False)
    endpoint['responses'] = [
        (200, {'choices': [{'finish_reason': 'length', 'message': {'content': '{"decision":'}}]}),
        (200, {'choices': [{'finish_reason': 'stop', 'message': {'content':
            '{"decision":"conflict","image_index":0,"reason":"Target has arms; this chair is armless."}'}}]}),
    ]
    result = verify_post_confirmation(ModelClient(), [Image.new('RGB', (8, 8))], 'chair',
                                      {'intrinsic_attributes': 'An armchair.'})
    assert result['decision'] == 'conflict' and len(endpoint['calls']) == 2


def test_post_transport_errors_use_the_same_three_attempt_budget(endpoint, monkeypatch):
    import requests
    from utils.instance_verifier import verify_post_confirmation
    monkeypatch.delenv('PSGNAV_LOG_FILE', raising=False)
    client = ModelClient()
    attempts = []

    def disconnected(*args, **kwargs):
        attempts.append(1)
        raise requests.exceptions.ChunkedEncodingError('premature response EOF')

    monkeypatch.setattr(client._session(), 'post', disconnected)
    with pytest.raises(ModelRequestError, match='ChunkedEncodingError'):
        verify_post_confirmation(client, [Image.new('RGB', (8, 8))], 'chair',
                                 {'intrinsic_attributes': 'An armchair.'})
    assert len(attempts) == 3


def _prompt_lock_bucket(client, prompt):
    payload = {
        "model": client.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": client.max_tokens,
        "stream": False,
        "enable_thinking": False,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return int(key[:8], 16) % len(client._key_locks)


def _two_unlocked_prompts(client):
    chosen = []
    buckets = set()
    index = 0
    while len(chosen) < 2:
        prompt = f'overlap-{index}'
        bucket = _prompt_lock_bucket(client, prompt)
        if bucket not in buckets:
            chosen.append(prompt)
            buckets.add(bucket)
        index += 1
    return chosen


def test_cache_hit_adds_no_model_time(endpoint):
    client = ModelClient()
    client.complete('cached-question')
    frozen = dict(client.stats)
    client.complete('cached-question')
    assert client.stats['requests'] == 1
    assert client.stats['cache_hits'] == 1
    assert client.stats['seconds'] == frozen['seconds']
    assert client.stats['text_seconds'] == frozen['text_seconds']
    assert client.stats['vision_seconds'] == 0


def test_overlapping_calls_record_one_real_interval(endpoint, monkeypatch):
    monkeypatch.setenv('PSGNAV_API_CONCURRENCY', '4')
    client = ModelClient()
    left, right = _two_unlocked_prompts(client)
    barrier = threading.Barrier(2)

    def send(_payload):
        barrier.wait(timeout=2)
        _REAL_SLEEP(0.25)
        return 'yes'

    monkeypatch.setattr(client, '_send', send)
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(client.complete, [left, right])) == ['yes', 'yes']
    assert client.stats['requests'] == 2
    assert client.stats['seconds'] >= 0.25
    assert client.stats['seconds'] < 0.40
    assert client.stats['text_seconds'] == pytest.approx(client.stats['seconds'])
    assert client.stats['vision_seconds'] == 0


def test_failed_call_still_records_real_time(endpoint, monkeypatch):
    client = ModelClient()

    def send(_payload):
        _REAL_SLEEP(0.05)
        raise ModelRequestError('down')

    monkeypatch.setattr(client, '_send', send)
    with pytest.raises(ModelRequestError, match='down'):
        client.complete('boom')
    assert client.stats['requests'] == 1
    assert client.stats['cache_hits'] == 0
    assert client.stats['seconds'] >= 0.05
    assert client.stats['text_seconds'] == pytest.approx(client.stats['seconds'])


def test_text_and_vision_real_time_stay_separate(endpoint, monkeypatch):
    client = ModelClient()

    def send(_payload):
        _REAL_SLEEP(0.05)
        return 'yes'

    monkeypatch.setattr(client, '_send', send)
    client.complete('words')
    text_seconds = client.stats['text_seconds']
    client.complete('picture', Image.new('RGB', (8, 8), 'red'))
    assert text_seconds >= 0.05
    assert client.stats['text_seconds'] == text_seconds
    assert client.stats['vision_seconds'] >= 0.05
    assert client.stats['seconds'] == pytest.approx(
        client.stats['text_seconds'] + client.stats['vision_seconds'])


def test_task_timing_snapshot_is_the_delta_across_clients(endpoint):
    from utils.model_client import diff_api_timing, format_api_timing, snapshot_api_timing
    before = snapshot_api_timing()
    client = ModelClient()
    client.complete('snapshot-task')
    delta = diff_api_timing(before, snapshot_api_timing())
    assert delta['requests'] == 1
    assert delta['cache_hits'] == 0
    assert delta['seconds'] == client.stats['seconds']
    assert delta['text_seconds'] == client.stats['text_seconds']
    line = format_api_timing(delta, gpu=0, dataset_index=3, episode_id=416)
    assert line == (
        '[API Timing] gpu=0 dataset_index=3 episode_id=416 '
        f'requests=1 cache_hits=0 seconds={delta["seconds"]:.3f} '
        f'text_seconds={delta["text_seconds"]:.3f} vision_seconds={delta["vision_seconds"]:.3f}'
    )
    assert 'gpu=' not in format_api_timing(
        delta, dataset_index=3, episode_id=416).split(' ', 2)[2]


def test_cached_reply_is_revalidated_for_a_stricter_caller(endpoint):
    from utils.instance_verifier import parse_matches
    endpoint['responses'] = [(200, {'choices':[{'message':{'content':'{"matches":["yes"]}'}}]})]
    client = ModelClient()
    client.complete('same')
    with pytest.raises(ValueError, match='wrong matches type or length'):
        client.complete('same', validator=lambda value:parse_matches(value,2))
    assert not client._cache
