import ast
from types import SimpleNamespace

import pytest
from habitat.core.benchmark import Benchmark


@pytest.mark.parametrize('mismatch', [False, True])
def test_resume_preserves_prefix_and_checks_identity(tmp_path, monkeypatch, mismatch):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path/'data/results_instancenav_test/experiment_0'
    folder.mkdir(parents=True)
    saved = dict(distance_to_goal=.5, success=1., spl=.8, softspl=.7,
                 episode_id='0', scene_id='scene', goal_object_id='1', goal='chair', num_steps=10)
    results = folder/'results.txt'
    results.write_text(repr(saved)+'\n')
    (folder/'results_0.txt').write_text(repr(dict(distance_to_goal=5))+'\n')
    (folder/'results_avg.txt').write_text(repr(dict(success=1.))+'\n')

    class Env:
        episodes = [0, 1]
        index = -1
        _max_episode_steps = 1000
        def reset(self):
            self.index += 1
            self.episode_over = False
            self.current_episode = SimpleNamespace(episode_id=str(self.index),
                scene_id='wrong' if mismatch else 'scene', goal_object_id='1')
            return {}
        def step(self, action):
            self.episode_over = True
            return {}
        def get_metrics(self):
            return dict(distance_to_goal=2., success=0., spl=0., softspl=0.)
    class Agent:
        config = SimpleNamespace(ENVIRONMENT=SimpleNamespace(ATOMIC_ACTIONS=False))
        args = SimpleNamespace(dataset='instancenav', resume=True)
        timestamp = 'test'
        experiment_name = 'experiment_0'
        obj_goal = 'chair'
        count_episodes = -1
        def reset(self):
            self.count_episodes += 1
            self.total_steps = 0
        def act(self, obs):
            self.total_steps += 1
            return {'action':0}
        def update_metrics(self, metrics):
            pass
    benchmark = Benchmark.__new__(Benchmark)
    benchmark._env = Env()
    benchmark._episode_start = 0
    benchmark._episode_end = None
    agent = Agent()
    if mismatch:
        with pytest.raises(ValueError, match='prefix mismatch'):
            benchmark.local_evaluate(agent)
        assert results.read_text() == repr(saved)+'\n'
    else:
        summary = benchmark.local_evaluate(agent)
        rows = [ast.literal_eval(line) for line in results.read_text().splitlines()]
        assert rows[0] == saved and len(rows) == 2
        assert rows[1]['episode_id'] == '1' and agent.count_episodes == 1
        assert summary['success'] == .5 and summary['spl'] == .4
