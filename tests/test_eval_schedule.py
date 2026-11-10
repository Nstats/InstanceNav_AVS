import ast
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from habitat.core.benchmark import Benchmark
from tools.eval_schedule import (
    claim_next, dual_progress_line, finish_episode, init_schedule,
    merge_worker_results, read_schedule, requeue_running, update_schedule,
)
from utils.model_client import ModelRequestError


def _done_metrics(**overrides):
    values = dict(measured_steps=1, measured_step_seconds=1.0, success=1.0, spl=0.5,
                  softspl=0.4, distance_to_goal=1.0, num_steps=10, goal='chair')
    values.update(overrides)
    return values


def test_claims_are_unique_and_in_order(tmp_path):
    path = tmp_path / 'schedule.json'
    init_schedule(path, 8)
    found = []
    found_lock = threading.Lock()

    def worker(gpu):
        while True:
            index = claim_next(path, gpu)
            if index is None:
                return
            with found_lock:
                found.append(index)
            finish_episode(path, gpu, index, **_done_metrics())

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    state = read_schedule(path)
    assert [item['dataset_index'] for item in state['assignments']] == list(range(8))
    assert sorted(found) == list(range(8))
    assert claim_next(path, 0) is None


def test_requeue_returns_only_interrupted_tasks(tmp_path):
    path = tmp_path / 'schedule.json'
    init_schedule(path, 5)
    assert claim_next(path, 0) == 0
    assert claim_next(path, 1) == 1
    finish_episode(path, 0, 0, **_done_metrics())
    state = requeue_running(path)
    assert state['pending'] == [1, 2, 3, 4]
    assert claim_next(path, 0) == 1
    done = [item['dataset_index'] for item in state['assignments'] if item['status'] == 'done']
    assert done == [0]


def test_progress_line_uses_completed_tasks_and_wall_clock():
    idle = {
        'total': 100,
        'max_episode_steps': 1000,
        'assignments': [],
        'workers': {},
    }
    assert dual_progress_line(idle, 10, [0, 1]) == (
        '进度 0/100 | 当前 gpu0:-- gpu1:-- | SR -- SPL -- | 已用 0:00:10 | 剩余约 --'
    )
    running = {
        'total': 4,
        'max_episode_steps': 1000,
        'assignments': [
            {'dataset_index': 0, 'gpu': 0, 'status': 'done', 'success': 1.0, 'spl': 0.5, 'num_steps': 100},
            {'dataset_index': 1, 'gpu': 1, 'status': 'done', 'success': 0.0, 'spl': 0.0, 'num_steps': 300},
        ],
        'workers': {
            '0': {'dataset_index': 2, 'current_steps': 50, 'measured_steps': 150},
            '1': {'dataset_index': 3, 'current_steps': 40, 'measured_steps': 40},
        },
    }
    assert dual_progress_line(running, 3800, [0, 1]) == (
        '进度 2/4 | 当前 gpu0:#2 50步 gpu1:#3 40步 | SR 0.500 SPL 0.250 | '
        '已用 1:03:20 | 剩余约 1.7 小时'
    )


class _QueueEnv:
    def __init__(self, episodes, queue):
        self.episodes = episodes
        self.queue = queue
        self._iter = iter(episodes)
        self._from_iter = True
        self._current = None
        self.episode_over = False
        self._max_episode_steps = 1000
        self._elapsed_steps = 0
        self.seen = []
        self.mid = None

    @property
    def current_episode(self):
        return self._current

    @current_episode.setter
    def current_episode(self, episode):
        self._current = episode
        self._from_iter = False

    def reset(self):
        if self._from_iter:
            self._current = next(self._iter)
        self._from_iter = True
        self.seen.append(self._current.index)
        self.episode_over = False
        self._elapsed_steps = 0
        return {}

    def step(self, _action):
        self._elapsed_steps += 1
        if self._elapsed_steps == 26 and self.mid is None:
            self.mid = read_schedule(self.queue)
        self.episode_over = self._elapsed_steps >= 26
        return {}

    def get_metrics(self):
        success = 1.0 if self._current.index % 2 == 0 else 0.0
        return dict(distance_to_goal=1.5, success=success, spl=0.25, softspl=0.4)


class _QueueAgent:
    def __init__(self, queue):
        self.config = SimpleNamespace(ENVIRONMENT=SimpleNamespace(ATOMIC_ACTIONS=False))
        self.args = SimpleNamespace(dataset='instancenav', resume=False, work_queue=str(queue), worker_gpu=0)
        self.timestamp = 'testjump'
        self.experiment_name = 'gpu0'
        self.obj_goal = 'chair'
        self.count_episodes = -1
        self.total_steps = 0

    def reset(self):
        self.count_episodes += 1
        self.total_steps = 0

    def act(self, _obs):
        self.total_steps += 1
        return {'action': 0}

    def update_metrics(self, _metrics):
        pass


def _episodes(count):
    return [SimpleNamespace(index=i, episode_id=str(100 + i), scene_id=f'scene{i}', goal_object_id=str(i))
            for i in range(count)]


def _run_queue(tmp_path, monkeypatch, episodes, edit=None, timestamp='testjump'):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('PSGNAV_LOG_FILE', raising=False)
    monkeypatch.delenv('PSGNAV_PROGRESS_FD', raising=False)
    queue = tmp_path / 'schedule.json'
    tasks = tmp_path / 'gpu0.tasks.json'
    monkeypatch.setenv('PSGNAV_TASKS_FILE', str(tasks))
    init_schedule(queue, len(episodes))
    if edit is not None:
        update_schedule(queue, edit)
    env = _QueueEnv(episodes, queue)
    benchmark = Benchmark.__new__(Benchmark)
    benchmark._env = env
    agent = _QueueAgent(queue)
    agent.timestamp = timestamp
    text = benchmark.local_evaluate(agent)
    output_path = tmp_path / f'data/results_instancenav_{timestamp}/gpu0/results.txt'
    rows = [ast.literal_eval(line) for line in output_path.read_text().splitlines()]
    summary = json.loads(tasks.read_text())
    return env, rows, summary, text


def test_worker_jumps_to_claimed_episode_and_records_tasks(tmp_path, monkeypatch, capsys):
    episodes = _episodes(3)

    def edit(state):
        state['pending'] = [2, 0, 1]

    env, rows, summary, averages = _run_queue(tmp_path, monkeypatch, episodes, edit)
    assert env.seen == [2, 0, 1]
    assert [row['dataset_index'] for row in rows] == [2, 0, 1]
    assert summary['dataset_indices'] == [0, 1, 2]
    assert summary['episode_ids'] == ['100', '101', '102']
    assert summary['execution_order'] == [2, 0, 1]
    assert env.mid['workers']['0']['dataset_index'] == 2
    assert env.mid['workers']['0']['current_steps'] == 25
    logged = capsys.readouterr().out
    assert logged.index('Episode 2:') < logged.index('Episode 0:') < logged.index('Episode 1:')
    assert '[Schedule] event=claim gpu=0 dataset_index=2 episode_id=102 scene_id=scene2 goal_object_id=2' in logged
    assert '[Schedule] event=summary gpu=0 count=3 dataset_indices=0,1,2 episode_ids=100,101,102' in logged
    assert averages['success'] == (1.0 + 1.0 + 0.0) / 3


class _ExhaustedAgent(_QueueAgent):
    def __init__(self, queue, *, exhausted, message='Model API HTTP 500 after 3 attempts'):
        super().__init__(queue)
        self._acts = 0
        self.exhausted = exhausted
        self.message = message

    def act(self, observations):
        self._acts += 1
        if self._acts == 27:
            raise ModelRequestError(self.message, retry_exhausted=self.exhausted)
        return super().act(observations)


def test_retry_exhaustion_defers_episode_and_resume_reclaims_it(tmp_path, monkeypatch, capsys):
    episodes = _episodes(3)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('PSGNAV_LOG_FILE', raising=False)
    monkeypatch.delenv('PSGNAV_PROGRESS_FD', raising=False)
    queue = tmp_path / 'schedule.json'
    tasks = tmp_path / 'gpu0.tasks.json'
    monkeypatch.setenv('PSGNAV_TASKS_FILE', str(tasks))
    init_schedule(queue, len(episodes))
    env = _QueueEnv(episodes, queue)
    benchmark = Benchmark.__new__(Benchmark)
    benchmark._env = env
    agent = _ExhaustedAgent(queue, exhausted=True)
    benchmark.local_evaluate(agent)
    output_path = tmp_path / 'data/results_instancenav_testjump/gpu0/results.txt'
    rows = [ast.literal_eval(line) for line in output_path.read_text().splitlines()]
    assert env.seen == [0, 1, 2]
    assert [row['dataset_index'] for row in rows] == [0, 2]
    state = read_schedule(queue)
    deferred = [item for item in state['assignments'] if item['status'] == 'deferred']
    assert [item['dataset_index'] for item in deferred] == [1]
    assert 'HTTP 500' in deferred[0]['reason']
    assert '[Schedule] event=deferred gpu=0 dataset_index=1 episode_id=101' in capsys.readouterr().out
    resumed = requeue_running(queue)
    assert resumed['pending'] == [1]
    assert claim_next(queue, 0) == 1
    done = [item['dataset_index'] for item in resumed['assignments'] if item['status'] == 'done']
    assert done == [0, 2]


def test_model_error_without_exhausted_retries_stops_the_worker(tmp_path, monkeypatch):
    episodes = _episodes(3)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('PSGNAV_LOG_FILE', raising=False)
    monkeypatch.delenv('PSGNAV_PROGRESS_FD', raising=False)
    queue = tmp_path / 'schedule.json'
    monkeypatch.setenv('PSGNAV_TASKS_FILE', str(tmp_path / 'gpu0.tasks.json'))
    init_schedule(queue, len(episodes))
    env = _QueueEnv(episodes, queue)
    benchmark = Benchmark.__new__(Benchmark)
    benchmark._env = env
    with pytest.raises(ModelRequestError, match='HTTP 401'):
        benchmark.local_evaluate(_ExhaustedAgent(
            queue, exhausted=False, message='Model API HTTP 401; check endpoint, key and model'))
    assert env.seen == [0, 1]
    state = read_schedule(queue)
    assert [item['dataset_index'] for item in state['assignments'] if item['status'] == 'done'] == [0]
    assert [item['dataset_index'] for item in state['assignments'] if item['status'] == 'running'] == [1]


def test_ordered_episodes_use_iterator_not_raw_list():
    class Dataset:
        def get_episode_iterator(self, **kwargs):
            assert kwargs['shuffle'] is False
            assert kwargs['seed'] == 77
            return iter(['grouped', 'next'])

    class Env:
        episodes = ['raw']
        _dataset = Dataset()
        _config = SimpleNamespace(
            ENVIRONMENT=SimpleNamespace(ITERATOR_OPTIONS={'SHUFFLE': False, 'CYCLE': False}),
            SEED=77,
        )

    benchmark = Benchmark.__new__(Benchmark)
    benchmark._env = Env()
    assert benchmark._ordered_eval_episodes() == ['grouped', 'next']


def test_resume_keeps_finished_rows_and_reruns_interrupted(tmp_path, monkeypatch, capsys):
    episodes = _episodes(3)
    finished = dict(distance_to_goal=0.2, success=1.0, spl=0.8, softspl=0.7,
                    episode_id='102', scene_id='scene2', goal_object_id='2', goal='chair',
                    num_steps=4, dataset_index=2)

    def edit(state):
        state['pending'] = [0, 1]
        state['assignments'].append({
            'dataset_index': 2, 'gpu': 0, 'status': 'done', 'episode_id': '102',
            'scene_id': 'scene2', 'goal_object_id': '2', 'success': 1.0, 'spl': 0.8,
            'softspl': 0.7, 'distance_to_goal': 0.2, 'num_steps': 4, 'goal': 'chair',
        })

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / 'data/results_instancenav_testjump/gpu0'
    folder.mkdir(parents=True)
    (folder / 'results.txt').write_text(repr(finished) + '\n')
    env, rows, summary, _averages = _run_queue(tmp_path, monkeypatch, episodes, edit)
    assert env.seen == [0, 1]
    assert [row['dataset_index'] for row in rows] == [2, 0, 1]
    assert rows[0] == finished
    assert summary['count'] == 3
    assert 'dataset_indices=0,1,2' in capsys.readouterr().out


def test_merge_sorts_by_dataset_index_and_reports_gaps(tmp_path):
    root = tmp_path / 'results'
    for gpu, indices in ((0, [0, 2]), (1, [1])):
        folder = root / f'gpu{gpu}'
        folder.mkdir(parents=True)
        rows = [dict(dataset_index=index, distance_to_goal=float(index), success=1.0, spl=0.1, softspl=0.2)
                for index in indices]
        (folder / 'results.txt').write_text(''.join(repr(row) + '\n' for row in rows))
    status = merge_worker_results(root, [0, 1], 3)
    assert status['missing_indices'] == [] and status['duplicate_indices'] == []
    merged = [ast.literal_eval(line) for line in (root / 'experiment_0/results.txt').read_text().splitlines()]
    assert [row['dataset_index'] for row in merged] == [0, 1, 2]
    averages = [ast.literal_eval(line) for line in (root / 'experiment_0/results_avg.txt').read_text().splitlines()]
    assert averages[-1]['success'] == 1.0
    duplicate_rows = [
        dict(dataset_index=1, distance_to_goal=1.0, success=1.0, spl=0.1, softspl=0.2),
        dict(dataset_index=0, distance_to_goal=9.0, success=0.0, spl=0.0, softspl=0.0),
    ]
    (root / 'gpu1/results.txt').write_text(''.join(repr(row) + '\n' for row in duplicate_rows))
    status = merge_worker_results(root, [0, 1], 3)
    assert status['duplicate_indices'] == [0]
    assert status['missing_indices'] == []
