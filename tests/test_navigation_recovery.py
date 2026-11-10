from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agent_test_support import load_agent_methods


AGENT_METHODS = ('act', 'reset_goal_detection', 'not_use_random_goal')


def make_agent(agent_class=None):
    agent_class = agent_class or load_agent_methods(*AGENT_METHODS)
    agent = agent_class.__new__(agent_class)
    agent.__dict__.update(
        args=SimpleNamespace(visualize=False), total_steps=30, max_episode_steps=1000,
        navigate_steps=8, obj_goal='chair', obj_goal_sg='chair',
        full_map=torch.zeros((1, 1, 40, 40)), full_pose=torch.tensor([1., 1., 0.]),
        map_size_cm=200, map_size=40, map_resolution=5, resolution=5,
        room_map=np.zeros((40, 40)), fbe_free_map=np.ones((40, 40)),
        goal_map=np.zeros((40, 40)), goal_gps_map=np.zeros((40, 40)),
        goal_gps_timestamp_map=np.zeros((40, 40)),
        goal_gps=np.array([.5, 0.]), possible_goal_temp_gps=np.array([.5, 0.]),
        found_goal=False, found_possible_goal=False, found_goal_times=0,
        last_gps=np.zeros(2), last_orientation=0., prev_action=1,
        move_steps=0, not_move_steps=0, forward_no_move_count=0,
        in_stuck_recovery=False, stuck_recovery_phase=None, stuck_recovery_target_angle=None,
        stuck_recovery_episode_count=0, orientation_history=[], orientation_history_size=10,
        history_pose=[], landmark_navigation_steps=0, max_landmark_navigation_steps=50,
        goal_loc=(20, 30), landmark_idx=0, unreachable_landmarks=set(),
        using_random_goal=False, move_since_random=0, random_this_ex=0, fronter_this_ex=0,
        random_nav_start_step=None, random_nav_periods=[],
    )
    agent.goal_map[20, 30] = 1
    state = dict(plans=[], plan_actions=deque(), default_action=1, frontiers=0,
                 frontier=(20, 30), random_goals=0, recovery_directions=0, resets=[])

    def noop(*args):
        pass

    agent.scenegraph = SimpleNamespace(**{name: noop for name in (
        'set_agent', 'set_navigate_steps', 'set_obj_goal', 'set_room_map',
        'set_fbe_free_map', 'set_observations', 'set_full_map', 'set_full_pose',
        'update_scenegraph', 'perception')})
    agent.update_map = agent.update_free_map = noop
    agent.get_traversible = lambda *args: (np.ones((40, 40)), (20, 20), 0)
    agent.snap_goal_to_traversible = lambda row, col, traversible: (row, col)
    agent._calculate_distance_to_goal = lambda: 1.
    agent.get_relative_goal_gps = lambda obs: np.zeros(2)

    def plan(traversible, goal_map, pose, start, start_o, found_goal):
        state['plans'].append((goal_map.copy(), found_goal))
        action = state['plan_actions'].popleft() if state['plan_actions'] else state['default_action']
        return 0, 0, False, action

    def frontier(*args):
        state['frontiers'] += 1
        return state['frontier'], 0

    def random_goal():
        state['random_goals'] += 1
        goal = np.zeros((40, 40))
        goal[10, 30] = 1
        return goal

    def open_direction(*args):
        state['recovery_directions'] += 1
        return None, None, 90.

    def reset_goal(reason='unspecified'):
        state['resets'].append(reason)
        agent_class.reset_goal_detection(agent, reason)

    agent._plan, agent.fbe, agent.set_random_goal = plan, frontier, random_goal
    agent.navigation_planner = SimpleNamespace(find_open_direction_for_stuck_recovery=open_direction)
    agent.reset_goal_detection = reset_goal
    observations = dict(gps=np.zeros(2), depth=np.ones((2, 2, 1)), rgb=np.zeros((2, 2, 3), dtype=np.uint8))
    return agent, state, observations


@pytest.mark.parametrize('locked', [False, True])
def test_recovery_keeps_its_direction_until_forward_even_with_empty_goal_map(locked):
    agent, state, obs = make_agent()
    agent.found_goal = locked
    agent.forward_no_move_count = agent.not_move_steps = 2
    actions = []
    for orientation in (0., 30., 60., 90., 90.):
        agent.full_pose[2] = orientation
        actions.append(agent.act(obs)['action'])
    assert actions == [2, 2, 2, 1, 1]
    assert state['recovery_directions'] == agent.stuck_recovery_episode_count == 1
    assert state['resets'] == ['stuck']
    assert len(state['plans']) == 1 and state['frontiers'] == 0
    assert agent.not_move_steps == 7  # FBE cannot erase the timeout during recovery.
    assert not agent.in_stuck_recovery and agent.stuck_recovery_phase is None


def test_recovery_runs_even_if_forward_failure_counter_has_cleared():
    agent, state, obs = make_agent()
    agent.in_stuck_recovery, agent.stuck_recovery_phase = True, 'turning'
    agent.stuck_recovery_target_angle = 90.
    agent.prev_action = 2
    assert agent.act(obs)['action'] == 2
    assert not state['plans'] and not state['resets'] and not state['frontiers']
    assert agent.stuck_recovery_target_angle == 90.


@pytest.mark.parametrize('recovering', [False, True])
@pytest.mark.parametrize('empty_goal', [False, True])
def test_long_stuck_timeout_plans_random_goal_instead_of_sending_stale_stop(recovering, empty_goal):
    agent, state, obs = make_agent()
    agent.not_move_steps = 12
    agent.forward_no_move_count = 3
    agent.in_stuck_recovery = recovering
    agent.stuck_recovery_phase = 'turning' if recovering else None
    agent.stuck_recovery_target_angle = 90. if recovering else None
    if empty_goal:
        agent.goal_map.fill(0)
    state['default_action'] = 3
    if not recovering and not empty_goal:
        state['plan_actions'].append(0)
    assert agent.act(obs)['action'] == 3
    assert state['random_goals'] == 1 and state['resets'] == ['stuck']
    assert agent.using_random_goal and not agent.in_stuck_recovery
    assert agent.not_move_steps == agent.forward_no_move_count == 0
    assert state['plans'][-1][0][10, 30] == 1


def test_timeout_random_goal_already_reached_uses_normal_waypoint_replanning():
    agent, state, obs = make_agent()
    agent.not_move_steps = 12
    agent.in_stuck_recovery, agent.stuck_recovery_phase = True, 'turning'
    agent.stuck_recovery_target_angle = 90.
    state['plan_actions'].extend([0, 2])
    assert agent.act(obs)['action'] == 2
    assert state['random_goals'] == state['frontiers'] == 1
    assert len(state['plans']) == 2


def test_real_progress_exits_recovery_and_resumes_normal_planning():
    agent, state, obs = make_agent()
    agent.in_stuck_recovery, agent.stuck_recovery_phase = True, 'turning'
    agent.stuck_recovery_target_angle = 90.
    agent.forward_no_move_count, agent.not_move_steps = 3, 8
    obs['gps'] = np.array([.1, 0.])
    assert agent.act(obs)['action'] == 1
    assert not agent.in_stuck_recovery and agent.stuck_recovery_phase is None
    assert agent.stuck_recovery_target_angle is None
    assert agent.not_move_steps == agent.forward_no_move_count == 0
    assert len(state['plans']) == 1 and not state['resets']


@pytest.mark.parametrize('mode', ['goal', 'possible', 'landmark', 'new_frontier', 'arrived', 'landmark_timeout', 'no_frontier'])
def test_ordinary_navigation_retains_goal_selection_and_actions(mode):
    agent, state, obs = make_agent()
    expected_frontiers, expected_plans, expected_random = 0, 1, 0
    state['default_action'] = 2
    if mode == 'goal':
        agent.found_goal = True
        state['default_action'] = 0  # Confirmed goals retain normal STOP behavior.
    elif mode == 'possible':
        agent.found_possible_goal = True
    elif mode == 'new_frontier':
        agent.goal_map.fill(0)
        expected_frontiers = 2  # Keep the baseline's initial FBE + waypoint-loop FBE.
    elif mode == 'arrived':
        state['plan_actions'].append(0)
        expected_frontiers, expected_plans = 1, 2
    elif mode == 'landmark_timeout':
        agent.landmark_navigation_steps = 50
        expected_frontiers, expected_plans = 1, 2
    elif mode == 'no_frontier':
        state['frontier'] = None
        state['plan_actions'].append(0)
        expected_frontiers, expected_plans, expected_random = 1, 2, 1
    assert agent.act(obs)['action'] == (0 if mode == 'goal' else 2)
    assert state['frontiers'] == expected_frontiers and len(state['plans']) == expected_plans
    assert state['random_goals'] == expected_random and not state['resets']
    assert not state['recovery_directions'] and not agent.in_stuck_recovery
