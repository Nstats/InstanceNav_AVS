"""Shared episode queue for one long-lived worker per GPU.

The pending list is claimed from the front, so GPU 0 and GPU 1 take
dataset indices 0 and 1 first, and whichever finishes takes the next
remaining index. A file lock keeps the two processes from taking the same task.
"""
import argparse
import ast
import fcntl
import json
import os
import signal
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


def _lock_path(path):
    path = Path(path)
    return path.parent / (path.name + '.lock')


@contextmanager
def _exclusive(path):
    lock_path = _lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, 'a+', encoding='utf-8') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def init_schedule(path, total, max_episode_steps=1000):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'schedule already exists: {path}')
    if total < 1:
        raise ValueError('total must be positive')
    state = {
        'total': int(total),
        'max_episode_steps': int(max_episode_steps),
        'pending': list(range(int(total))),
        'assignments': [],
        'workers': {},
        'watch_stop': False,
    }
    with _exclusive(path):
        if path.exists():
            raise FileExistsError(f'schedule already exists: {path}')
        _write(path, state)
    return state


def read_schedule(path):
    path = Path(path)
    with _exclusive(path):
        return _read(path)


def update_schedule(path, editor):
    path = Path(path)
    with _exclusive(path):
        state = _read(path)
        editor(state)
        _write(path, state)
        return state


def _worker_slot(state, gpu):
    workers = state.setdefault('workers', {})
    slot = workers.get(str(gpu), {})
    return workers, slot


def claim_next(path, gpu):
    """Pop the next dataset index, or return None when the queue is empty."""
    path = Path(path)
    gpu = int(gpu)
    with _exclusive(path):
        state = _read(path)
        pending = state['pending']
        if not pending:
            return None
        index = int(pending.pop(0))
        state['assignments'].append({
            'dataset_index': index,
            'gpu': gpu,
            'status': 'running',
            'episode_id': None,
            'scene_id': None,
            'goal_object_id': None,
        })
        workers, previous = _worker_slot(state, gpu)
        workers[str(gpu)] = {
            'dataset_index': index,
            'current_steps': 0,
            'measured_steps': int(previous.get('measured_steps', 0)),
            'measured_step_seconds': float(previous.get('measured_step_seconds', 0.0)),
            'pid': os.getpid(),
        }
        _write(path, state)
        return index


def note_episode(path, gpu, dataset_index, *, episode_id, scene_id, goal_object_id):
    path = Path(path)
    gpu = int(gpu)
    dataset_index = int(dataset_index)

    def edit(state):
        assignment = _running(state, gpu, dataset_index)
        assignment['episode_id'] = None if episode_id is None else str(episode_id)
        assignment['scene_id'] = None if scene_id is None else str(scene_id)
        assignment['goal_object_id'] = None if goal_object_id is None else str(goal_object_id)

    update_schedule(path, edit)


def heartbeat(path, gpu, *, dataset_index, current_steps, measured_steps, measured_step_seconds):
    path = Path(path)
    gpu = int(gpu)

    def edit(state):
        workers, _previous = _worker_slot(state, gpu)
        workers[str(gpu)] = {
            'dataset_index': None if dataset_index is None else int(dataset_index),
            'current_steps': int(current_steps),
            'measured_steps': int(measured_steps),
            'measured_step_seconds': float(measured_step_seconds),
            'pid': os.getpid(),
        }

    update_schedule(path, edit)


def finish_episode(path, gpu, dataset_index, *, measured_steps, measured_step_seconds,
                    success, spl, softspl, distance_to_goal, num_steps, goal):
    path = Path(path)
    gpu = int(gpu)
    dataset_index = int(dataset_index)

    def edit(state):
        assignment = _running(state, gpu, dataset_index)
        assignment.update({
            'status': 'done',
            'success': float(success),
            'spl': float(spl),
            'softspl': float(softspl),
            'distance_to_goal': float(distance_to_goal),
            'num_steps': int(num_steps),
            'goal': None if goal is None else str(goal),
        })
        workers, _previous = _worker_slot(state, gpu)
        workers[str(gpu)] = {
            'dataset_index': None,
            'current_steps': 0,
            'measured_steps': int(measured_steps),
            'measured_step_seconds': float(measured_step_seconds),
            'pid': os.getpid(),
        }

    update_schedule(path, edit)


def defer_episode(path, gpu, dataset_index, *, measured_steps, measured_step_seconds, reason):
    """Leave a claimed task unfinished after retry exhaustion. A later --resume claims it again."""
    path = Path(path)
    gpu = int(gpu)
    dataset_index = int(dataset_index)

    def edit(state):
        assignment = _running(state, gpu, dataset_index)
        assignment.update({
            'status': 'deferred',
            'reason': str(reason)[:500],
        })
        workers, _previous = _worker_slot(state, gpu)
        workers[str(gpu)] = {
            'dataset_index': None,
            'current_steps': 0,
            'measured_steps': int(measured_steps),
            'measured_step_seconds': float(measured_step_seconds),
            'pid': os.getpid(),
        }

    update_schedule(path, edit)


def set_max_episode_steps(path, max_episode_steps):
    def edit(state):
        state['max_episode_steps'] = int(max_episode_steps)

    update_schedule(path, edit)


def requeue_running(path):
    """Return interrupted and deferred tasks to the pending list, in index order."""
    path = Path(path)
    with _exclusive(path):
        state = _read(path)
        returned = []
        for assignment in state['assignments']:
            if assignment.get('status') in {'running', 'deferred'}:
                assignment['status'] = 'requeued'
                returned.append(int(assignment['dataset_index']))
        pending = [int(index) for index in state['pending']]
        for index in returned:
            if index not in pending:
                pending.append(index)
        state['pending'] = sorted(set(pending))
        state['workers'] = {}
        state['watch_stop'] = False
        _write(path, state)
        return state


def request_watch_stop(path):
    def edit(state):
        state['watch_stop'] = True

    update_schedule(path, edit)


def _running(state, gpu, dataset_index):
    for assignment in reversed(state['assignments']):
        if (assignment.get('status') == 'running'
                and int(assignment['gpu']) == int(gpu)
                and int(assignment['dataset_index']) == int(dataset_index)):
            return assignment
    raise KeyError(f'no running assignment gpu={gpu} dataset_index={dataset_index}')


def latest_assignments(state):
    latest = {}
    for assignment in state['assignments']:
        latest[int(assignment['dataset_index'])] = assignment
    return latest


def done_assignments(state, gpu=None):
    rows = []
    for assignment in latest_assignments(state).values():
        if assignment.get('status') != 'done':
            continue
        if gpu is not None and int(assignment['gpu']) != int(gpu):
            continue
        rows.append(assignment)
    rows.sort(key=lambda item: int(item['dataset_index']))
    return rows


def format_elapsed(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f'{hours}:{minutes:02d}:{secs:02d}'


def dual_progress_line(state, elapsed_seconds, gpus):
    """One terminal line: completed count, SR/SPL, elapsed wall time, and ETA."""
    total = int(state['total'])
    done = done_assignments(state)
    completed = len(done)
    elapsed_seconds = max(0.0, float(elapsed_seconds))
    if completed:
        success = sum(float(item['success']) for item in done) / completed
        spl = sum(float(item['spl']) for item in done) / completed
        scores = f'SR {success:.3f} SPL {spl:.3f}'
        mean_steps = sum(int(item['num_steps']) for item in done) / completed
    else:
        scores = 'SR -- SPL --'
        mean_steps = int(state.get('max_episode_steps', 1000))
    workers = state.get('workers', {})
    measured = 0
    running_steps = 0
    current = []
    for gpu in gpus:
        info = workers.get(str(gpu), {})
        measured += int(info.get('measured_steps', 0))
        index = info.get('dataset_index', None)
        steps = int(info.get('current_steps') or 0)
        if index is None:
            label = '空闲' if info else '--'
            current.append(f'gpu{gpu}:{label}')
        else:
            current.append(f'gpu{gpu}:#{int(index)} {steps}步')
            running_steps += steps
    unfinished = total - completed
    remaining_steps = max(0.0, unfinished * mean_steps - running_steps)
    if measured > 0 and elapsed_seconds > 0:
        eta = f'{(remaining_steps / (measured / elapsed_seconds)) / 3600:.1f} 小时'
    else:
        eta = '--'
    return (
        f'进度 {completed}/{total} | 当前 {" ".join(current)} | {scores} | '
        f'已用 {format_elapsed(elapsed_seconds)} | 剩余约 {eta}'
    )


def worker_summary(state, gpu):
    rows = done_assignments(state, gpu)
    execution = [int(item['dataset_index']) for item in state['assignments']
                 if item.get('status') == 'done' and int(item['gpu']) == int(gpu)]
    tasks = [{
        'dataset_index': int(item['dataset_index']),
        'episode_id': item.get('episode_id'),
        'scene_id': item.get('scene_id'),
        'goal_object_id': item.get('goal_object_id'),
        'success': item.get('success'),
        'spl': item.get('spl'),
        'num_steps': item.get('num_steps'),
    } for item in rows]
    return {
        'gpu': int(gpu),
        'count': len(rows),
        'dataset_indices': [int(item['dataset_index']) for item in rows],
        'episode_ids': [item.get('episode_id') for item in rows],
        'execution_order': execution,
        'tasks': tasks,
    }


def summary_line(summary):
    indices = ','.join(str(index) for index in summary['dataset_indices'])
    episode_ids = ','.join('' if item is None else str(item) for item in summary['episode_ids'])
    return (
        f'[Schedule] event=summary gpu={summary["gpu"]} count={summary["count"]} '
        f'dataset_indices={indices} episode_ids={episode_ids}'
    )


def write_tasks_file(schedule_path, gpu, tasks_path):
    summary = worker_summary(read_schedule(schedule_path), gpu)
    path = Path(tasks_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)
    return summary


def _load_result_rows(folder):
    results = Path(folder) / 'results.txt'
    if not results.exists():
        return []
    return [ast.literal_eval(line) for line in results.read_text(encoding='utf-8').splitlines() if line.strip()]


def merge_worker_results(results_root, gpus, total):
    """Write experiment_0 results in dataset_index order. Return missing and duplicate indices."""
    root = Path(results_root)
    by_index = {}
    duplicates = []
    for gpu in gpus:
        for row in _load_result_rows(root / f'gpu{gpu}'):
            index = int(row['dataset_index'])
            if index in by_index:
                duplicates.append(index)
            by_index[index] = row
    missing = [index for index in range(int(total)) if index not in by_index]
    ordered = [by_index[index] for index in range(int(total)) if index in by_index]
    destination = root / 'experiment_0'
    destination.mkdir(parents=True, exist_ok=True)
    _write_result_lines(destination / 'results.txt', ordered)
    totals = defaultdict(float)
    averages = []
    for count, row in enumerate(ordered, start=1):
        for name in ('distance_to_goal', 'success', 'spl', 'softspl'):
            totals[name] += float(row[name])
        averages.append({name: value / count for name, value in totals.items()})
    _write_result_lines(destination / 'results_avg.txt', averages)
    status = {
        'total': int(total),
        'completed': len(ordered),
        'missing_indices': missing,
        'duplicate_indices': sorted(set(duplicates)),
    }
    (destination / 'merge_status.json').write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    return status


def _write_result_lines(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with open(temporary, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write('%s\n' % row)
    os.replace(temporary, path)


def _parse_gpus(text):
    gpus = [int(item) for item in text.split(',') if item != '']
    if len(gpus) != 2 or len(set(gpus)) != 2 or any(gpu < 0 for gpu in gpus):
        raise ValueError('--gpus expects two distinct non-negative ids, such as 0,1')
    return gpus


class _ProgressDisplay:
    def __init__(self):
        self._stream = None
        self._clear = ''
        raw = os.environ.get('PSGNAV_PROGRESS_FD', '').strip()
        if not raw:
            self._stream = sys.stdout
            self._clear = '\033[K' if sys.stdout.isatty() else ''
            return
        fd = os.dup(int(raw))
        self._stream = os.fdopen(fd, 'w', buffering=1)
        self._clear = '\033[K' if os.isatty(fd) else ''

    def render(self, line):
        self._stream.write('\r' + line + self._clear)
        self._stream.flush()

    def finish(self):
        if self._stream is None:
            return
        self._stream.write('\n')
        self._stream.flush()
        if self._stream is not sys.stdout:
            self._stream.close()
        self._stream = None


def watch_progress(path, gpus, started):
    display = _ProgressDisplay()
    stop = {'value': False}

    def handle_stop(_signum, _frame):
        stop['value'] = True

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    try:
        while not stop['value']:
            try:
                state = read_schedule(path)
            except (OSError, json.JSONDecodeError):
                time.sleep(0.2)
                continue
            elapsed = time.time() - float(started)
            display.render(dual_progress_line(state, elapsed, gpus))
            if state.get('watch_stop'):
                break
            time.sleep(1.0)
    finally:
        display.finish()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Dual-GPU InstanceNav schedule')
    commands = parser.add_subparsers(dest='command', required=True)

    init_cmd = commands.add_parser('init')
    init_cmd.add_argument('--schedule', required=True)
    init_cmd.add_argument('--total', required=True, type=int)
    init_cmd.add_argument('--max-episode-steps', default=1000, type=int)

    requeue_cmd = commands.add_parser('requeue')
    requeue_cmd.add_argument('--schedule', required=True)

    watch_cmd = commands.add_parser('watch')
    watch_cmd.add_argument('--schedule', required=True)
    watch_cmd.add_argument('--gpus', required=True)
    watch_cmd.add_argument('--started', required=True, type=float)

    stop_cmd = commands.add_parser('stop-watch')
    stop_cmd.add_argument('--schedule', required=True)

    merge_cmd = commands.add_parser('merge')
    merge_cmd.add_argument('--results-root', required=True)
    merge_cmd.add_argument('--gpus', required=True)
    merge_cmd.add_argument('--total', required=True, type=int)

    args = parser.parse_args(argv)
    if args.command == 'init':
        init_schedule(args.schedule, args.total, args.max_episode_steps)
        return 0
    if args.command == 'requeue':
        state = requeue_running(args.schedule)
        print(f'Requeued running tasks. pending={len(state["pending"])} total={state["total"]}')
        return 0
    if args.command == 'watch':
        watch_progress(args.schedule, _parse_gpus(args.gpus), args.started)
        return 0
    if args.command == 'stop-watch':
        request_watch_stop(args.schedule)
        return 0
    status = merge_worker_results(args.results_root, _parse_gpus(args.gpus), args.total)
    print(
        f'Merged {status["completed"]}/{status["total"]} '
        f'missing={status["missing_indices"]} duplicates={status["duplicate_indices"]}'
    )
    if status['missing_indices'] or status['duplicate_indices']:
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
