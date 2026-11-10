"""Conditional locked-goal continuations using the production PSG controller.

Only start where logged approach state and movement determine the controller's
reset counters. Other locks are explicitly excluded, not initialized by guessing.
No exploration or language model call is allowed in a continuation.
"""
import copy
from types import SimpleNamespace
import numpy as np
import torch
import skimage.morphology

from PSG_Nav import PSG_Nav_Agent
from utils.map_manager import MapManager
from utils.navigation_planner import NavigationPlanner
from utils.utils_fmm.mapping import Semantic_Mapping
from configs.dataset_categories import dataset_to_unified_name
from tools.coin_calibration_nodes import sg_name


class BranchEnded(Exception):
    pass


class LockedGraph:
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return lambda *args, **kwargs: None


class LockedAgent(PSG_Nav_Agent):
    def update_free_map(self, observations):
        # Dead work in this isolated replay: _plan/get_traversible consume only
        # obstacle, visited and collision maps. Any recovery/exploration that
        # could read the free-space map terminates at the overrides below.
        # Production PSG_Nav_Agent is unchanged.
        pass

    def reset_goal_detection(self, reason='unspecified'):
        raise BranchEnded('unlock_' + reason)

    def fbe(self, *args):
        raise BranchEnded('exploration_boundary')


def make_agent(config, geometry):
    agent = LockedAgent.__new__(LockedAgent)
    agent.config, agent.coin_geometry = config.clone(), geometry
    agent.device = torch.device(f'cuda:{torch.cuda.current_device()}')
    agent.dataset = 'coin'
    agent.map_resolution = agent.resolution = 5
    agent.map_size_cm, agent.map_size = 4000, 800
    agent.dilation_deg, agent.collision_threshold = 0, .08
    agent.selem = skimage.morphology.square(1)
    agent.map_manager = MapManager(agent)
    agent.map_manager.init_map()
    agent.sem_map_module = Semantic_Mapping(agent, camera_height_cm=131, fov=79).to(agent.device).eval()
    agent.free_map_module = Semantic_Mapping(agent, max_height=10, min_height=-150,
                                           camera_height_cm=131, fov=79).to(agent.device).eval()
    agent.navigation_planner = NavigationPlanner(agent)
    agent.scenegraph, agent.args = LockedGraph(), SimpleNamespace(visualize=False, real_world_playback=False)
    agent.relative_evidence_controller = agent.detection_manager = None
    agent.max_episode_steps, agent.max_landmark_navigation_steps = 500, 50
    agent.total_steps = agent.navigate_steps = agent.move_steps = 0
    agent.prev_action = agent.former_collide = agent.not_move_steps = agent.forward_no_move_count = 0
    agent.last_loc = agent.full_pose.clone()
    agent.last_gps = np.array([11100., 11100.])
    agent.last_orientation = 0.
    agent.landmark_navigation_steps = 1
    agent.in_stuck_recovery = False
    agent.stuck_recovery_phase = agent.stuck_recovery_target_angle = None
    agent.using_random_goal, agent.move_since_random = False, 0
    agent.random_nav_start_step, agent.random_nav_periods = None, []
    agent.history_pose, agent.orientation_history = [], []
    agent.orientation_history_size = 10
    agent.found_goal, agent.found_possible_goal = True, False
    agent.count_episodes = 0
    return agent


class StopReplay:
    def __init__(self, config, geometry):
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = True
        self.config, self.geometry = config, geometry

    def begin(self, item, ep, renderer):
        self.item, self.ep, self.renderer = item, ep, renderer
        self.agent = make_agent(self.config, self.geometry)
        self.agent.obj_goal = dataset_to_unified_name(ep.object_category, 'coin')
        self.agent.obj_goal_sg = sg_name(ep.object_category)
        self.previous = None
        self.branches, self.excluded = [], []
        self.last_lock_step = max(item['locks'], default=0)

    def frame(self, step, obs, atomic, sim):
        from tools.coin_calibration import observations
        # The outer replay still audits every action and observation. No future
        # continuation can consume maps after the final recorded lock.
        if step['policy_step'] > self.last_lock_step:
            return
        agent = self.agent
        prior = self.previous
        moved = np.linalg.norm(obs['gps'] - agent.last_gps) >= .08
        locks = self.item['locks'].get(step['policy_step'], [])
        for lock in locks:
            # The preceding approach frame sets landmark_navigation_steps to 1;
            # current translational progress resets all stuck/collision counters.
            certified = (prior is not None and prior['policy_step'] >= 23 and moved and
                         prior['found_possible_goal'] and not prior['in_stuck_recovery'] and
                         not prior['using_random_goal'])
            if not certified:
                self.excluded.append({'frame_id': lock['frame_id'], 'step': step['policy_step'],
                                      'reason': 'controller_state_not_recoverable_from_log'})
                continue
            saved = sim.get_agent_state()
            cached_path = self.ep._shortest_path_cache
            frame = next(f for f in self.item['frames'][step['policy_step']] if f['frame_id'] == lock['frame_id'])
            from tools.build_verification_set import label_box
            for value in self.renderer.info.values():
                value['goal_category'] = value['name'].strip().lower()
            label = label_box(np.asarray(obs['semantic']).squeeze(), obs['depth'].squeeze(),
                              lock['box'], int(self.ep.goal_object_id), self.renderer.info,
                              self.ep.object_category)
            for metric, radius in [('fmm', .25), ('fmm', .5), ('fmm', .75), ('fmm', 1.), ('euclidean', .75)]:
                branch = copy.deepcopy(agent)
                branch.config.defrost()
                branch.config.TASK.STOP_DISTANCE_METRIC = metric
                branch.config.TASK.STOP_DISTANCE = radius
                branch.config.freeze()
                branch.goal_gps = np.array(lock['goal_gps'])
                branch.total_steps = step['policy_step'] - 1
                branch.landmark_navigation_steps = 1
                branch.sem_map_module.view_angles = list(frame['mapping_view_angles_deg'])
                branch.free_map_module.view_angles = list(frame['mapping_view_angles_deg'])
                sim.get_agent(0).set_state(saved, reset_sensors=False, infer_sensor_states=False)
                self.ep._shortest_path_cache = None
                from habitat.tasks.nav.nav import DistanceToGoal
                distance = DistanceToGoal(sim, self.config.TASK.DISTANCE_TO_GOAL)
                distance.reset_metric(self.ep)
                used, collisions, reason, stop_distance = 0, 0, 'budget', None
                while atomic + used < 500:
                    current, _, _ = observations(sim, self.ep, self.geometry)
                    # act expects raw sensor depth; observations has prepared it.
                    current['depth'] = current.pop('depth_wide')
                    current.pop('semantic', None)
                    try:
                        with torch.no_grad():
                            action = branch.act(current)['action']
                    except BranchEnded as exc:
                        reason = str(exc)
                        break
                    used += 1
                    if action == 0:
                        reason, stop_distance = 'stop', float(distance.get_metric())
                        break
                    if action not in (1, 2, 3):
                        raise RuntimeError(f'Unexpected locked-goal action {action}')
                    sim.step(action)
                    collisions += int(sim.previous_step_collided)
                    distance.update_metric(self.ep)
                self.branches.append({'frame_id': lock['frame_id'], 'step': step['policy_step'],
                    'navigation_track_id': lock['navigation_track_id'], 'metric': metric, 'radius_m': radius,
                    'target_label': label, 'reason': reason, 'success': bool(reason == 'stop' and stop_distance < .25),
                    'stop_dtg_m': stop_distance, 'extra_actions': used, 'collisions': collisions})
            sim.get_agent(0).set_state(saved, reset_sensors=False, infer_sensor_states=False)
            self.ep._shortest_path_cache = cached_path
        # Rebuild the obstacle/free/visited maps from EVERY recorded observation,
        # including frames with no GLIP candidates. The path is forced by the log.
        angle = 0 if step['policy_step'] == 1 else (30 if step['policy_step'] <= 8 else
                 60 if step['policy_step'] <= 15 else 30 if step['policy_step'] == 16 else 0)
        agent.sem_map_module.set_view_angles(angle)
        agent.free_map_module.set_view_angles(angle)
        with torch.no_grad():
            agent.update_map(obs)
            agent.update_free_map(obs)
        pose = np.zeros(7)
        pose[:3] = agent.full_pose.cpu().numpy()
        pose[1], pose[2], pose[4], pose[6] = 40 - pose[1], -pose[2], 800, 800
        agent.get_traversible(agent.full_map.cpu().numpy()[0, 0, ::-1], pose)
        if prior is not None and prior['action'] == 1:
            # Production _plan marks the same collision rays independently of
            # which exploration goal was selected. Reconstruct those writes.
            x1, y1, t1 = agent.last_loc.cpu().numpy()
            x2, y2, _ = agent.full_pose.cpu().numpy()
            if np.hypot(x2 - x1, y2 - y1) < .08:
                for i in range(5):
                    wx = x1 + .05 * (i + 4) * np.cos(np.deg2rad(-t1))
                    wy = 40 - y1 + .05 * (i + 4) * np.sin(np.deg2rad(-t1))
                    r, c = np.clip([round(wy * 20), round(wx * 20)], 0, 799).astype(int)
                    agent.collision_map[r, c] = 1
        agent.last_loc = agent.full_pose.clone()
        agent.last_gps = obs['gps'].copy()
        agent.last_orientation = float(agent.full_pose[2])
        agent.prev_action = step['action']
        agent.total_steps = step['policy_step']
        self.previous = step

    def finish(self):
        return {'branches': self.branches, 'excluded_locks': self.excluded,
                'continuation_kind': 'conditional_locked_goal_only',
                'eligibility': 'previous approach, no recovery/random goal, current movement >= 0.08m',
                'stop_schema': 2,
                'branch_count': len(self.branches)}
