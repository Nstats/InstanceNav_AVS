"""--gpus accepts one GPU as well as several; the queue, logs and merge are unchanged."""
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tools.eval_schedule import _parse_gpus, dual_progress_line, init_schedule, merge_worker_results, read_schedule

ROOT = Path(__file__).resolve().parents[1]

FAKE = '''
import os, sys, time
from pathlib import Path
args = sys.argv[1:]
with open('calls.txt', 'a') as handle:
    handle.write(os.environ.get('CUDA_VISIBLE_DEVICES', '-') + ' ' + ' '.join(args) + '\\n')
if args == ['-']:
    sys.stdin.read()
elif '-c' in args:
    print(time.time())
elif 'PSG_Nav.py' in args:
    gpu = args[args.index('--worker-gpu') + 1]
    sys.exit(int(os.environ.get('TEST_EXIT_' + gpu, '0')))
elif 'watch' in args:
    while not Path('watch.stopped').exists():
        time.sleep(.02)
elif 'stop-watch' in args:
    Path('watch.stopped').touch()
'''


def launch(tmp_path, gpus, *options, **extra):
    script = tmp_path / 'scripts/run_instance100.sh'
    script.parent.mkdir(exist_ok=True)
    script.write_text((ROOT / 'scripts/run_instance100.sh').read_text())
    fake = tmp_path / 'python'
    fake.write_text(f'#!{sys.executable}\n' + FAKE)
    fake.chmod(0o755)
    env = {**os.environ, 'PSGNAV_PYTHON': str(fake), **extra}
    env.pop('ENABLE_CONTRADICTION_CHECK', None)
    process = subprocess.Popen(['bash', str(script), '--gpus', gpus, '--timestamp', 'test', *options],
                               cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=15)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    calls = (tmp_path / 'calls.txt').read_text().splitlines() if (tmp_path / 'calls.txt').exists() else []
    return process.returncode, stdout, stderr, calls


def test_single_gpu_runs_one_worker_on_that_gpu(tmp_path):
    code, stdout, stderr, calls = launch(tmp_path, '2')
    assert code == 0, stdout + stderr
    workers = [line for line in calls if 'PSG_Nav.py' in line]
    assert len(workers) == 1
    assert workers[0].startswith('2 ') and '--worker-gpu 2' in workers[0]
    assert any('watch' in line and '--gpus 2' in line for line in calls)
    assert any(' merge ' in f' {line} ' and '--gpus 2' in line for line in calls)
    assert 'GPUs: 2 (1 worker(s))' in stdout and 'gpu2 exit 0' in stdout
    assert (tmp_path / 'outputs/instancenav_val10_100_test_gpu2.log').exists()


def test_full_val_propagates_to_both_workers_and_merge(tmp_path):
    from tools.run_instance_ablation import FULL_CONFIG, launch_command, settings
    import yaml
    base = yaml.safe_load((ROOT / 'configs/runtime/instancenav_unseen10_1000atomic_1m.yaml').read_text())
    config = yaml.safe_load((ROOT / FULL_CONFIG).read_text())
    base['DATASET']['DATA_PATH'] = 'data/datasets/instancenav/val_full/val_full.json.gz'
    assert config == base  # Dataset size, not geometry or metrics, is the sole protocol change.
    args = launch_command(settings('A0', nav_base='vlfm'), '0,1', full=True)
    code, out, err, calls = launch(tmp_path, '0,1', *args[2:])
    assert code == 0, out + err
    workers = [line.split() for line in calls if 'VLFM_Nav.py' in line]
    assert {w[w.index('--worker-gpu') + 1] for w in workers} == {'0', '1'}
    for w in workers:
        assert w[w.index('--config') + 1] == FULL_CONFIG
        assert w[w.index('--episode_end') + 1] == '1000'
    for action in ('init', 'merge'):
        assert any(f' {action} ' in line and '--total 1000' in line for line in calls)


def test_single_gpu_failure_is_reported(tmp_path):
    code, stdout, stderr, _ = launch(tmp_path, '0', TEST_EXIT_0='7')
    assert code != 0 and 'gpu0 exit 7' in stdout and 'Evaluation incomplete.' in stdout


def test_three_gpus_run_three_workers(tmp_path):
    code, stdout, stderr, calls = launch(tmp_path, '0,1,3')
    assert code == 0, stdout + stderr
    assert sorted(line.split()[0] for line in calls if 'PSG_Nav.py' in line) == ['0', '1', '3']


@pytest.mark.parametrize('gpus', ['', '0,0', 'a', '0,', '-1'])
def test_invalid_gpu_lists_fail_before_python(tmp_path, gpus):
    code, _, _, calls = launch(tmp_path, gpus)
    assert code == 2 and not calls


def test_existing_results_from_another_gpu_set_block_a_fresh_start(tmp_path):
    (tmp_path / 'data/results_instancenav_test/gpu1').mkdir(parents=True)
    code, _, stderr, calls = launch(tmp_path, '0')
    assert code == 1 and 'Results already exist' in stderr
    assert not any('PSG_Nav.py' in line for line in calls)


def test_schedule_tools_accept_one_gpu(tmp_path):
    assert _parse_gpus('0') == [0] and _parse_gpus('1,0') == [1, 0]
    for bad in ('', '0,0', '-1'):
        with pytest.raises(ValueError):
            _parse_gpus(bad)
    schedule = tmp_path / 'schedule.json'
    init_schedule(schedule, 3)
    assert dual_progress_line(read_schedule(schedule), 0, [0]).startswith('进度 0/3 | 当前 gpu0:--')
    status = merge_worker_results(tmp_path, [0], 3)
    assert status['completed'] == 0 and status['missing_indices'] == [0, 1, 2]


def test_merge_keeps_tasks_from_an_earlier_gpu_set(tmp_path):
    # Started on 0,1, resumed on 0 alone: gpu1's finished tasks still count.
    row = "{'dataset_index': %d, 'distance_to_goal': 1.0, 'success': 1.0, 'spl': 1.0, 'softspl': 1.0}\n"
    for gpu, indices in ((0, [0]), (1, [1, 2])):
        (tmp_path / f'gpu{gpu}').mkdir()
        (tmp_path / f'gpu{gpu}/results.txt').write_text(''.join(row % i for i in indices))
    status = merge_worker_results(tmp_path, [0], 3)
    assert status['completed'] == 3 and not status['missing_indices'] and not status['duplicate_indices']


@pytest.mark.parametrize('gpus,ok', [('0', True), ('0,1', True), ('0,1,2', True),
                                     ('0,0', False), ('a', False), ('', False)])
def test_ablation_launcher_accepts_one_or_more_gpus(gpus, ok):
    result = subprocess.run([sys.executable, str(ROOT / 'tools/run_instance_ablation.py'), 'A0',
                             '--gpus', gpus, '--timestamp', 'probe', '--suite', 'probe', '--dry-run'],
                            cwd=ROOT, capture_output=True, text=True)
    assert (result.returncode == 0) == ok, result.stderr
    if ok:
        assert f"'--gpus', '{gpus}'" in result.stdout or f'"{gpus}"' in result.stdout
