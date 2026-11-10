import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from utils.instance_evidence import PositiveEvidenceStore
from utils.instance_verifier import parse_post_confirmation_check, verify_post_confirmation
from utils.model_client import ModelRequestError


@pytest.mark.parametrize('answer', [
    '{"decision":"unknown"}',
    '{"decision":"yes","image_index":-1,"reason":""}',
    '{"decision":"conflict","image_index":-1,"reason":"Blue seat."}',
    '{"decision":"conflict","image_index":2,"reason":"Blue seat."}',
    '{"decision":"conflict","image_index":true,"reason":"Blue seat."}',
    '{"decision":"conflict","image_index":0,"reason":" "}',
    '{"decision":"no_conflict","image_index":0,"reason":""}',
    '{"decision":"conflict","decision":"no_conflict","image_index":-1,"reason":""}',
    '{"decision":"unknown","image_index":-1,"reason":"","extra":true}',
    '[{"decision":"unknown","image_index":-1,"reason":""},{}]',
])
def test_invalid_post_decisions_cannot_be_interpreted_as_passes(answer):
    with pytest.raises(ValueError):
        parse_post_confirmation_check(answer, 2)


def test_unambiguous_json_wrappers_are_accepted():
    value = {'decision': 'conflict', 'image_index': 1, 'reason': 'The target has arms; this chair is armless.'}
    assert parse_post_confirmation_check('```json\n' + json.dumps([value]) + '\n```', 2) == value


def test_three_format_failures_print_reason_and_preserve_exact_input_evidence(tmp_path, monkeypatch, capsys):
    logfile = tmp_path / 'evaluation.log'
    monkeypatch.setenv('PSGNAV_LOG_FILE', str(logfile))
    calls = []
    images = [Image.new('RGB', (16, 16), color) for color in ('red', 'blue')]

    def complete(prompt, **options):
        calls.append((prompt, options))
        assert options['images'] == images and options['retries'] == 0
        options['validator']('{"decision":"no_conflict"}')

    with pytest.raises(ModelRequestError, match='failed after 3 attempts'):
        verify_post_confirmation(SimpleNamespace(complete=complete), images, 'chair',
                                 {'intrinsic_attributes': 'A red chair.'})
    assert len(calls) == 3
    captured = capsys.readouterr()
    assert 'expected exactly decision, image_index, reason' in captured.err
    prefix = '[Instance Verification] '
    records = [json.loads(line[len(prefix):]) for line in captured.out.splitlines() if line.startswith(prefix)]
    assert [record['event'] for record in records] == ['request'] + ['raw_response', 'format_error'] * 3 + ['failed']
    assert len({record['verification_id'] for record in records}) == 1
    assert records[-1]['attempts'] == 3 and len(records[-1]['errors']) == 3
    evidence = Path(records[0]['evidence_dir'])
    saved = json.loads((evidence / 'input.json').read_text())
    assert saved['context']['image_count'] == 2
    for name, expected in zip(saved['images'], images):
        with Image.open(evidence / name) as actual:
            assert actual.tobytes() == expected.tobytes()


def test_geometrically_ambiguous_tracks_and_different_graph_nodes_do_not_mix():
    def candidate(x, node=None, box=None):
        return {'gps': np.array([x, 0.]), 'node_id': node, 'box': box or [0, 0, 10, 10]}

    store = PositiveEvidenceStore(.8)
    left, right = store.associate([candidate(0), candidate(.4, box=[20, 0, 30, 10])], 0)
    ambiguous, = store.associate([candidate(.2)], 1)
    assert ambiguous not in (left, right)
    first_node, = store.associate([candidate(0, node=10)], 2)
    second_node, = store.associate([candidate(0, node=11)], 3)
    assert first_node is not second_node
    seen_again, = store.associate([candidate(2, node=10)], 4)
    assert seen_again is first_node


@pytest.mark.parametrize('failed_gpu', ['0', '1', 'none'])
def test_dual_launcher_stops_the_sibling_when_either_worker_fails(tmp_path, failed_gpu):
    # Run an isolated copy with fake evaluators: no real GPU process is started or signaled.
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / 'scripts/run_instance100.sh'
    script.parent.mkdir()
    script.write_text((root / 'scripts/run_instance100.sh').read_text())
    fake = tmp_path / 'python'
    fake.write_text(f'#!{sys.executable}\n' + '''
import os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
if args == ['-']:
    sys.stdin.read()
elif '-c' in args:
    print(time.time())
elif 'PSG_Nav.py' in args:
    gpu = args[args.index('--worker-gpu') + 1]
    def stopped(*_):
        Path('worker_' + gpu + '.stopped').touch()
        sys.exit(143)
    signal.signal(signal.SIGTERM, stopped)
    Path('worker_' + gpu + '.started').touch()
    deadline = time.monotonic() + 8
    while not all(Path('worker_' + g + '.started').exists() for g in ('0', '1')):
        if time.monotonic() > deadline:
            sys.exit(99)
        time.sleep(.02)
    failed = os.environ['TEST_FAILED_GPU']
    if failed == 'none':
        sys.exit(0)
    if gpu == failed:
        print('Post-confirmation contradiction check failed after 3 attempts; format error', flush=True)
        sys.exit(7)
    while True:
        time.sleep(.05)
elif 'watch' in args:
    while not Path('watch.stopped').exists():
        time.sleep(.02)
elif 'stop-watch' in args:
    Path('watch.stopped').touch()
''')
    fake.chmod(0o755)
    env = {**os.environ, 'PSGNAV_PYTHON': str(fake), 'TEST_FAILED_GPU': failed_gpu,
           'INSTANCE_CONFIRMATION_MODE': 'positive_match',
           'ENABLE_INSTANCE_CONFIRMATION': 'true',
           'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY': 'true',
           'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK': 'true'}
    env.pop('ENABLE_CONTRADICTION_CHECK', None)
    process = subprocess.Popen(['bash', str(script), '--gpus', '0,1', '--timestamp', 'test'],
                               cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=15)
    finally:
        # Clean up the isolated process group even if the assertion/timeout reveals a regression.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    if failed_gpu == 'none':
        assert process.returncode == 0, stdout + stderr
    else:
        assert process.returncode != 0
        sibling = '1' if failed_gpu == '0' else '0'
        assert (tmp_path / f'worker_{sibling}.stopped').exists(), stdout + stderr
        log = tmp_path / f'outputs/instancenav_val10_100_test_gpu{failed_gpu}.log'
        assert 'failed after 3 attempts' in log.read_text()
        assert 'stopping both workers' in stderr
    for gpu in ('0', '1'):
        log = tmp_path / f'outputs/instancenav_val10_100_test_gpu{gpu}.log'
        assert 'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY: true' in log.read_text()
