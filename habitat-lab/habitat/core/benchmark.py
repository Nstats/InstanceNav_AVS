#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""Implements evaluation of ``habitat.Agent`` inside ``habitat.Env``.
``habitat.Benchmark`` creates a ``habitat.Env`` which is specified through
the ``config_env`` parameter in constructor. The evaluation is task agnostic
and is implemented through metrics defined for ``habitat.EmbodiedTask``.
"""

import io
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, Optional
from datetime import datetime
from habitat.config.default import get_config
from habitat.core.agent import Agent
from habitat.core.env import Env
from habitat.core.logging import logger
import habitat_sim.agent
import json
import ast
import time
import imageio
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision.utils import save_image


class _TeeStream(io.TextIOBase):
    """Startup output goes to the terminal and the log until navigation begins."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        first = self._streams[0]
        return first.isatty() if hasattr(first, "isatty") else False


def _navigation_log_path():
    path = os.environ.get("PSGNAV_LOG_FILE", "").strip()
    return path or None


def _retarget_habitat_logger(stream):
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setStream(stream)


_startup_mirror = {}


def mirror_startup_logs():
    """Copy process text output to PSGNAV_LOG_FILE without hiding the terminal."""
    path = _navigation_log_path()
    if not path or _startup_mirror.get("on") or not sys.stdout.isatty():
        return
    log = open(path, "a", buffering=1, encoding="utf-8")
    _startup_mirror.update(on=True, stdout=sys.stdout, stderr=sys.stderr, log=log)
    sys.stdout = _TeeStream(sys.stdout, log)
    sys.stderr = _TeeStream(sys.stderr, log)
    _retarget_habitat_logger(sys.stderr)


def redirect_navigation_logs():
    """Send further process output only to the log. The progress fd stays put."""
    path = _navigation_log_path()
    if not path:
        return
    if _startup_mirror.get("on"):
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = _startup_mirror["stdout"]
        sys.stderr = _startup_mirror["stderr"]
        _startup_mirror["log"].flush()
    sys.stdout.flush()
    sys.stderr.flush()
    log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
    finally:
        os.close(log_fd)
    _retarget_habitat_logger(sys.stderr)


def _print_api_timing(agent, dataset_index, episode_id, gpu=None):
    """Log this task's real model-call time. Agents without the hook stay silent."""
    report = getattr(agent, "api_timing_delta", None)
    if report is None:
        return
    from utils.model_client import format_api_timing
    print(
        format_api_timing(report(), gpu=gpu, dataset_index=dataset_index, episode_id=episode_id),
        flush=True,
    )


def _format_elapsed(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


class _TerminalProgress:
    """One refreshing status line on PSGNAV_PROGRESS_FD. Unset leaves the terminal alone."""

    def __init__(self):
        self._stream = None
        self._clear = ""
        raw = os.getenv("PSGNAV_PROGRESS_FD")
        if not raw:
            return
        try:
            fd = os.dup(int(raw))
            self._stream = os.fdopen(fd, "w", buffering=1)
            self._clear = "\033[K" if os.isatty(fd) else ""
        except (OSError, ValueError):
            self._stream = None

    def render(self, progress, scored_episodes, agg_metrics):
        if self._stream is None:
            return
        completed = progress["completed_episodes"]
        total = progress["total_episodes"]
        steps = progress["current_episode_steps"]
        if scored_episodes:
            success = agg_metrics["success"] / scored_episodes
            spl = agg_metrics["spl"] / scored_episodes
            scores = f"SR {success:.3f} SPL {spl:.3f}"
            remaining = progress.get(
                "remaining_hours_at_completed_episode_mean",
                progress.get("remaining_hours_if_all_run_to_limit"),
            )
        else:
            scores = "SR -- SPL --"
            remaining = progress.get("remaining_hours_if_all_run_to_limit")
        elapsed = _format_elapsed(progress["elapsed_seconds"])
        eta = "--" if remaining is None else f"{remaining:.1f} 小时"
        line = (
            f"进度 {completed}/{total} | 当前 {steps} 步 | {scores} | "
            f"已用 {elapsed} | 剩余约 {eta}"
        )
        self._stream.write("\r" + line + self._clear)
        self._stream.flush()

    def finish(self):
        if self._stream is None:
            return
        self._stream.write("\n")
        self._stream.flush()
        self._stream.close()
        self._stream = None


try:
    from utils.log_utils import log_event as _diag_event
except ImportError:  # benchmark used outside the PSG-Nav repo
    import json as _json

    def _diag_event(tag, **fields):
        print(f'[{tag}] ' + _json.dumps(fields, ensure_ascii=True, default=str),
              flush=True)


class _GoalVisibilityTracker:
    """Ground-truth goal visibility from the semantic sensor.

    Logging only: observations are never fed back into the policy. The pixel
    id is resolved against the scene semantic annotations once per episode so
    an id-encoding mismatch shows up in the log instead of silently reading 0.
    """

    def __init__(self, env, goal_object_id, *, gpu=None, dataset_index=None):
        self._env = env
        self._gpu = gpu
        self._dataset_index = dataset_index
        self.target_id = None
        self.pixel_id = None
        self.mapping_found = None
        self.goal_world_pos = None
        self.seen_steps = 0
        self.max_pixels = 0
        self.last_visible = None
        self.last_pixels = 0
        try:
            self.target_id = int(str(goal_object_id))
        except (TypeError, ValueError):
            self.target_id = None
        self._resolve(goal_object_id)

    def _resolve(self, goal_object_id):
        if self.target_id is None:
            self.mapping_found = False
        else:
            self.pixel_id = self.target_id
            try:
                objects = self._env.sim.semantic_annotations().objects
                match = None
                for obj in objects:
                    if str(getattr(obj, 'id', '')) == str(goal_object_id):
                        match = obj
                        break
                if match is None:
                    for obj in objects:
                        if getattr(obj, 'semantic_id', None) == self.target_id:
                            match = obj
                            break
                self.mapping_found = match is not None
                if match is not None:
                    semantic_id = getattr(match, 'semantic_id', None)
                    if semantic_id is not None:
                        self.pixel_id = int(semantic_id)
                    aabb = getattr(match, 'aabb', None)
                    if aabb is not None and hasattr(aabb, 'center'):
                        self.goal_world_pos = [round(float(v), 3) for v in aabb.center]
            except Exception as exc:  # mapping check is best-effort diagnostics
                self.mapping_found = None
                logger.warning('[GoalVisibility] semantic mapping check failed: %s', exc)
        _diag_event('Goal Mapping', gpu=self._gpu, dataset_index=self._dataset_index,
                    goal_object_id=(str(goal_object_id) if goal_object_id is not None else None),
                    pixel_id=self.pixel_id, mapping_found=self.mapping_found,
                    goal_world_pos=self.goal_world_pos)

    def update(self, observations):
        """Refresh visibility from the latest observation (once per policy step)."""
        sem = observations.get('semantic') if isinstance(observations, dict) else None
        if sem is None or self.pixel_id is None:
            self.last_visible = None
            self.last_pixels = 0
            return
        count = int((np.asarray(sem) == self.pixel_id).sum())
        self.last_pixels = count
        self.last_visible = count > 0
        if count > 0:
            self.seen_steps += 1
            self.max_pixels = max(self.max_pixels, count)


def _agent_pose_fields(agent, prefix=''):
    fields = {}
    pose = getattr(agent, 'full_pose', None)
    if pose is not None:
        try:
            values = pose.cpu().numpy() if hasattr(pose, 'cpu') else np.asarray(pose)
            fields[prefix + 'agent_world'] = [round(float(values[0]), 3),
                                              round(float(values[1]), 3)]
            fields[prefix + 'heading_deg'] = round(float(values[2]), 1)
        except Exception:
            pass
    map_pos = getattr(agent, 'agent_map_pos', None)
    if map_pos is not None:
        fields[prefix + 'agent_map'] = [int(map_pos[0]), int(map_pos[1])]
    return fields


def _locked_goal_map(agent):
    goal_map = getattr(agent, 'goal_map', None)
    if goal_map is None:
        return None
    cells = np.argwhere(goal_map == 1)
    if len(cells) == 0:
        return None
    return [int(cells[0][0]), int(cells[0][1])]


def _log_policy_step(*, env, agent, action, executed, metrics, tracker,
                     gpu=None, dataset_index=None):
    fields = dict(gpu=gpu, dataset_index=dataset_index,
                  episode=getattr(agent, 'count_episodes', None),
                  policy_step=getattr(agent, 'total_steps', None),
                  atomic_steps=getattr(env, '_elapsed_steps', None),
                  action=(action.get('action') if isinstance(action, dict) else action),
                  primitives=executed,
                  found_goal=bool(getattr(agent, 'found_goal', False)),
                  found_possible_goal=bool(getattr(agent, 'found_possible_goal', False)),
                  in_stuck_recovery=bool(getattr(agent, 'in_stuck_recovery', False)),
                  using_random_goal=bool(getattr(agent, 'using_random_goal', False)),
                  locked_goal_map=_locked_goal_map(agent),
                  dtg_internal=(round(float(agent.distance_to_goal), 3)
                                if getattr(agent, 'distance_to_goal', None) is not None else None),
                  dtg_gt=(round(float(metrics['distance_to_goal']), 3)
                          if metrics and metrics.get('distance_to_goal') is not None else None))
    fields.update(_agent_pose_fields(agent))
    if tracker is not None and tracker.last_visible is not None:
        fields['goal_visible'] = int(tracker.last_visible)
        fields['goal_visible_px'] = tracker.last_pixels
    _diag_event('Step', **{k: v for k, v in fields.items() if v is not None})


def _log_episode_end(*, env, agent, raw, tracker, termination, num_steps,
                     gpu=None, dataset_index=None):
    random_steps = 0
    for start, end in getattr(agent, 'random_nav_periods', []) or []:
        random_steps += max(0, int(end) - int(start))
    fields = dict(gpu=gpu, dataset_index=dataset_index,
                  episode=getattr(agent, 'count_episodes', None),
                  termination=termination,
                  num_steps=num_steps,
                  num_policy_steps=getattr(agent, 'total_steps', None),
                  success=raw.get('success'), spl=raw.get('spl'),
                  softspl=raw.get('softspl'),
                  final_dtg_gt=(round(float(raw['distance_to_goal']), 3)
                                if raw.get('distance_to_goal') is not None else None),
                  found_goal=bool(getattr(agent, 'found_goal', False)),
                  found_goal_times=int(getattr(agent, 'found_goal_times', 0) or 0),
                  locked_goal_map=_locked_goal_map(agent),
                  stuck_recoveries=getattr(agent, 'stuck_recovery_episode_count', None),
                  random_nav_steps=random_steps)
    fields.update(_agent_pose_fields(agent, prefix='final_'))
    if tracker is not None:
        fields.update(goal_mapping_found=tracker.mapping_found,
                      goal_world_pos=tracker.goal_world_pos,
                      goal_seen_steps=tracker.seen_steps,
                      goal_max_visible_px=tracker.max_pixels)
        if tracker.last_visible is not None:
            fields['final_goal_visible'] = int(tracker.last_visible)
            fields['final_goal_visible_px'] = tracker.last_pixels
    _diag_event('EpisodeEnd', **{k: v for k, v in fields.items() if v is not None})


def _termination_reason(env, last_policy_action):
    """Why the episode ended: explicit STOP, budget exhaustion, or other."""
    elapsed = getattr(env, '_elapsed_steps', 0) or 0
    max_steps = getattr(env, '_max_episode_steps', None)
    if max_steps is not None and elapsed >= max_steps:
        return 'step_limit'
    if last_policy_action == 0:
        return 'stop_called'
    return 'other'


class Benchmark:
    r"""Benchmark for evaluating agents in environments."""

    def __init__(
        self, config_paths: Optional[str] = None, eval_remote: bool = False, split_l: int = -1, split_r: int = -1, episode_start: int = 0, episode_end: Optional[int] = None
    ) -> None:
        r"""..

        :param config_paths: file to be used for creating the environment
        :param eval_remote: boolean indicating whether evaluation should be run remotely or locally
        :param split_l: starting index for scene split
        :param split_r: ending index for scene split
        :param episode_start: starting episode index within each scene
        :param episode_end: ending episode index within each scene (None means all episodes)
        """
        config_env = get_config(config_paths)
        config_env.DATASET.defrost()
        config_env.DATASET.SPLIT_L = split_l
        config_env.DATASET.SPLIT_R = split_r
        config_env.DATASET.freeze()
        self._eval_remote = eval_remote
        self._episode_start = episode_start
        self._episode_end = episode_end

        if self._eval_remote is True:
            self._env = None
        else:
            self._env = Env(config=config_env)

    def remote_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ):
        # The modules imported below are specific to habitat-challenge remote evaluation.
        # These modules are not part of the habitat-lab repository.
        import pickle
        import time

        import evalai_environment_habitat  # noqa: F401
        import evaluation_pb2
        import evaluation_pb2_grpc
        import grpc

        def pack_for_grpc(entity):
            return pickle.dumps(entity)

        def unpack_for_grpc(entity):
            return pickle.loads(entity)

        def remote_ep_over(stub):
            res_env = unpack_for_grpc(
                stub.episode_over(evaluation_pb2.Package()).SerializedEntity
            )
            return res_env["episode_over"]

        env_address_port = os.environ.get("EVALENV_ADDPORT", "localhost:8085")
        channel = grpc.insecure_channel(env_address_port)
        stub = evaluation_pb2_grpc.EnvironmentStub(channel)

        base_num_episodes = unpack_for_grpc(
            stub.num_episodes(evaluation_pb2.Package()).SerializedEntity
        )
        num_episodes = base_num_episodes["num_episodes"]

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0

        while count_episodes < num_episodes:
            agent.reset()
            res_env = unpack_for_grpc(
                stub.reset(evaluation_pb2.Package()).SerializedEntity
            )

            while not remote_ep_over(stub):
                obs = res_env["observations"]
                action = agent.act(obs)

                res_env = unpack_for_grpc(
                    stub.act_on_environment(
                        evaluation_pb2.Package(
                            SerializedEntity=pack_for_grpc(action)
                        )
                    ).SerializedEntity
                )

            metrics = unpack_for_grpc(
                stub.get_metrics(
                    evaluation_pb2.Package(
                        SerializedEntity=pack_for_grpc(action)
                    )
                ).SerializedEntity
            )

            for m, v in metrics["metrics"].items():
                agg_metrics[m] += v
            count_episodes += 1

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        stub.evalai_update_submission(evaluation_pb2.Package())

        return avg_metrics

    def local_evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        work_queue = getattr(getattr(agent, 'args', None), 'work_queue', None)
        if work_queue:
            return self._evaluate_work_queue(agent, work_queue)
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        # Apply episode range filtering
        episode_end = self._episode_end if self._episode_end is not None else num_episodes
        episode_end = min(episode_end, num_episodes)

        assert episode_end > self._episode_start, \
            f"episode_end ({episode_end}) must be greater than episode_start ({self._episode_start})"

        num_episodes_to_run = episode_end - self._episode_start
        print(f"Running episodes {self._episode_start} to {episode_end-1} (total: {num_episodes_to_run} episodes)")

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0  # Total episodes processed (including skipped)
        executed_episodes = 0  # Episodes actually executed
        all_metrics = []
        all_metrics_0 = []
        all_metrics_avg = []
        count_success = 0
        agent.simulator = self
        run_started = time.perf_counter()
        measured_step_seconds = 0.0
        measured_steps = 0
        completed_episode_steps = 0
        progress_timestamp = getattr(agent, 'timestamp', datetime.now().strftime('%m%d_%H%M'))
        progress_dataset = getattr(getattr(agent, 'args', None), 'dataset', '')
        progress_suffix = f'_{progress_dataset}' if progress_dataset else ''
        progress_dir = Path(f'data/results{progress_suffix}_{progress_timestamp}/{agent.experiment_name}')
        progress_dir.mkdir(parents=True, exist_ok=True)
        resume_rows = []
        if getattr(getattr(agent, 'args', None), 'resume', False):
            if self._episode_start != 0:
                raise ValueError('--resume requires episode_start=0 to verify the saved prefix')
            results_file = progress_dir / 'results.txt'
            if not results_file.exists():
                raise ValueError('--resume requires an existing results.txt and the original timestamp')
            resume_rows = [ast.literal_eval(line) for line in results_file.read_text().splitlines() if line.strip()]
            if len(resume_rows) > episode_end:
                raise ValueError('Saved results exceed the requested episode range')
            all_metrics = list(resume_rows)
            for row in resume_rows:
                for name in ('distance_to_goal', 'success', 'spl', 'softspl'):
                    agg_metrics[name] += row[name]
                completed_episode_steps += row['num_steps']
            executed_episodes = len(resume_rows)
            count_success = sum(row['success'] for row in resume_rows)
            for filename, output in [('results_0.txt', all_metrics_0), ('results_avg.txt', all_metrics_avg)]:
                file = progress_dir / filename
                if file.exists():
                    output.extend(ast.literal_eval(line) for line in file.read_text().splitlines()[:executed_episodes])
            agent.count_episodes = executed_episodes - 1
            print(f'[Resume] Preserving {executed_episodes} completed episodes; restarting interrupted episode from its initial state', flush=True)
        elif (progress_dir / 'results.txt').exists():
            raise ValueError('Results already exist; choose a new timestamp or pass --resume')

        terminal = _TerminalProgress()
        redirect_navigation_logs()
        terminal.render(
            {
                "completed_episodes": executed_episodes,
                "total_episodes": num_episodes_to_run,
                "current_episode_steps": 0,
                "elapsed_seconds": time.perf_counter() - run_started,
            },
            executed_episodes,
            agg_metrics,
        )
        try:
            while count_episodes < episode_end:
                observations = self._env.reset()

                if count_episodes < len(resume_rows):
                    saved = resume_rows[count_episodes]
                    episode = self._env.current_episode
                    if (str(saved['episode_id']), saved['scene_id'], str(saved['goal_object_id'])) != (
                            str(episode.episode_id), episode.scene_id, str(getattr(episode, 'goal_object_id', None))):
                        raise ValueError(f'Resume dataset prefix mismatch at episode {count_episodes}')
                    count_episodes += 1
                    continue

                # Skip episodes before episode_start
                if count_episodes < self._episode_start:
                    count_episodes += 1
                    continue

                # Stop if we've reached episode_end
                if count_episodes >= episode_end:
                    break

                agent.reset()
                metrics = self._env.get_metrics()
                all_metrics_0.append(metrics)
                pending_terminal = None
                tracker = _GoalVisibilityTracker(
                    self._env, getattr(self._env.current_episode, 'goal_object_id', None))
                last_policy_action = None

                while not self._env.episode_over:
                    step_started = time.perf_counter()
                    action = agent.act(observations)
                    atomic_actions = bool(getattr(agent.config.ENVIRONMENT, 'ATOMIC_ACTIONS', False))
                    from utils.atomic_actions import expand_atomic_actions
                    executed_actions = 0
                    for primitive in expand_atomic_actions(action, atomic_actions):
                        observations = self._env.step(primitive)
                        executed_actions += 1
                        if self._env.episode_over:
                            break
                    episode_steps = self._env._elapsed_steps if atomic_actions else agent.total_steps
                    current_metrics = self._env.get_metrics()
                    agent.update_metrics(current_metrics)
                    tracker.update(observations)
                    _log_policy_step(env=self._env, agent=agent, action=action,
                                     executed=executed_actions, metrics=current_metrics,
                                     tracker=tracker)
                    last_policy_action = action.get('action') if isinstance(action, dict) else action
                    measured_step_seconds += time.perf_counter() - step_started
                    measured_steps += executed_actions
                    if episode_steps % 25 == 0 or self._env.episode_over:
                        current_done = int(self._env.episode_over)
                        mean_step = measured_step_seconds / measured_steps
                        episodes_left = num_episodes_to_run - executed_episodes - current_done
                        max_steps_left = episodes_left * self._env._max_episode_steps
                        if not current_done:
                            max_steps_left -= episode_steps
                        progress = {
                            'completed_episodes': executed_episodes + current_done,
                            'total_episodes': num_episodes_to_run,
                            'current_episode_id': self._env.current_episode.episode_id,
                            'current_episode_steps': episode_steps,
                            'step_unit': 'atomic_action' if atomic_actions else 'policy_action',
                            'elapsed_seconds': time.perf_counter() - run_started,
                            'measured_steps': measured_steps,
                            'seconds_per_step': mean_step,
                            'remaining_hours_if_all_run_to_limit': max_steps_left * mean_step / 3600,
                            'estimate_note': 'Measured throughput extrapolation, not a completion guarantee.',
                        }
                        finished_count = executed_episodes + current_done
                        if finished_count:
                            finished_steps = completed_episode_steps + (episode_steps if current_done else 0)
                            progress['mean_completed_episode_steps'] = finished_steps / finished_count
                            expected_steps = episodes_left * progress['mean_completed_episode_steps']
                            if not current_done:
                                expected_steps = max(0, expected_steps - episode_steps)
                            progress['remaining_hours_at_completed_episode_mean'] = expected_steps * mean_step / 3600
                        temporary = progress_dir / 'progress.json.tmp'
                        temporary.write_text(json.dumps(progress, indent=2))
                        temporary.replace(progress_dir / 'progress.json')
                        print(f'[Progress] completed={progress["completed_episodes"]}/{num_episodes_to_run} '
                              f'episode_steps={episode_steps} seconds_per_step={mean_step:.3f}', flush=True)
                        if self._env.episode_over:
                            pending_terminal = progress
                        else:
                            terminal.render(progress, executed_episodes, agg_metrics)

                metrics = dict(self._env.get_metrics())
                metrics['episode_id'] = self._env.current_episode.episode_id
                metrics['scene_id'] = self._env.current_episode.scene_id
                metrics['goal_object_id'] = getattr(self._env.current_episode, 'goal_object_id', None)
                metrics['goal'] = (self._env.current_episode.object_category
                                   if getattr(agent, 'dataset', None) == 'coin' else agent.obj_goal)
                metrics['num_steps'] = episode_steps
                if atomic_actions:
                    metrics['num_policy_steps'] = agent.total_steps
                    metrics['step_unit'] = 'atomic_action'
                _log_episode_end(env=self._env, agent=agent, raw=metrics,
                                 tracker=tracker,
                                 termination=_termination_reason(self._env, last_policy_action),
                                 num_steps=episode_steps)
                all_metrics.append(metrics)
                metrics = self._env.get_metrics()
                print(f"Episode {count_episodes}: {metrics}")
                _print_api_timing(
                    agent, count_episodes, self._env.current_episode.episode_id,
                    gpu=getattr(getattr(agent, "args", None), "worker_gpu", None),
                )
                if metrics['success'] == 1:
                    count_success += 1
                for m, v in metrics.items():
                    if isinstance(v, dict):
                        for sub_m, sub_v in v.items():
                            agg_metrics[m + "/" + str(sub_m)] += sub_v
                    else:
                        agg_metrics[m] += v

                count_episodes += 1
                executed_episodes += 1
                completed_episode_steps += episode_steps
                avg_metrics = {k: v / executed_episodes for k, v in agg_metrics.items()}
                all_metrics_avg.append(avg_metrics)
                for k,v in avg_metrics.items():
                    logger.info("{}: {}".format(k, v))
            
                experiment_name = agent.experiment_name

                # Get timestamp from agent (unified across parallel runs)
                # If agent doesn't have timestamp, generate one (backward compatibility)
                if hasattr(agent, 'timestamp'):
                    timestamp = agent.timestamp
                else:
                    timestamp = datetime.now().strftime("%m%d_%H%M")

                # Add dataset suffix and timestamp to results directory
                # Format: results_mp3d_1220_1430/experiment_0/
                dataset_suffix = f"_{agent.args.dataset}" if hasattr(agent, 'args') and hasattr(agent.args, 'dataset') and agent.args.dataset else ""
                path = f'data/results{dataset_suffix}_{timestamp}/{experiment_name}'
                if not os.path.exists(path):
                    os.makedirs(path)

                with open(f'{path}/results.txt.tmp', 'w') as fp:
                    for item in all_metrics:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                os.replace(f'{path}/results.txt.tmp', f'{path}/results.txt')

                with open(f'{path}/results_0.txt', 'w') as fp:
                    for item in all_metrics_0:
                        # write each item on a new line
                        fp.write("%s\n" % item)

                with open(f'{path}/results_avg.txt', 'w') as fp:
                    for item in all_metrics_avg:
                        # write each item on a new line
                        fp.write("%s\n" % item)
                if pending_terminal is not None:
                    terminal.render(pending_terminal, executed_episodes, agg_metrics)



        finally:
            terminal.finish()

        avg_metrics = {k: v / executed_episodes for k, v in agg_metrics.items()}

        return avg_metrics

    def _ordered_eval_episodes(self):
        """Episode order used by a fresh single-process run, including scene grouping."""
        env = self._env
        dataset = getattr(env, '_dataset', None)
        config = getattr(env, '_config', None)
        if dataset is not None and config is not None:
            options = {
                key.lower(): value
                for key, value in config.ENVIRONMENT.ITERATOR_OPTIONS.items()
            }
            options['seed'] = config.SEED
            return list(dataset.get_episode_iterator(**options))
        return list(env.episodes)

    def _evaluate_work_queue(self, agent: "Agent", work_queue) -> Dict[str, float]:
        """Claim dataset indices until the shared queue is empty. One process stays alive."""
        import signal
        from tools.eval_schedule import (
            defer_episode, finish_episode, heartbeat, note_episode, claim_next,
            read_schedule, set_max_episode_steps, summary_line, write_tasks_file,
        )
        from utils.model_client import ModelRequestError

        gpu = getattr(agent.args, 'worker_gpu', None)
        if gpu is None or int(gpu) < 0:
            raise ValueError('--work-queue requires a non-negative --worker-gpu')
        gpu = int(gpu)
        queue_path = Path(work_queue)
        ordered = self._ordered_eval_episodes()
        schedule = read_schedule(queue_path)
        if int(schedule['total']) > len(ordered):
            raise ValueError(
                f'queue total {schedule["total"]} exceeds available episodes {len(ordered)}'
            )
        set_max_episode_steps(queue_path, int(self._env._max_episode_steps))
        progress_timestamp = getattr(agent, 'timestamp', datetime.now().strftime('%m%d_%H%M'))
        progress_dataset = getattr(getattr(agent, 'args', None), 'dataset', '')
        progress_suffix = f'_{progress_dataset}' if progress_dataset else ''
        progress_dir = Path(f'data/results{progress_suffix}_{progress_timestamp}/{agent.experiment_name}')
        progress_dir.mkdir(parents=True, exist_ok=True)
        results_file = progress_dir / 'results.txt'
        all_metrics = []
        if results_file.exists():
            all_metrics = [ast.literal_eval(line) for line in results_file.read_text().splitlines() if line.strip()]
        agent.simulator = self
        redirect_navigation_logs()
        print(f'Work queue: {queue_path}', flush=True)
        print(f'Worker GPU: {gpu}', flush=True)
        print(f'Ordered episodes: {len(ordered)}', flush=True)
        measured_step_seconds = 0.0
        measured_steps = 0

        def stop_on_term(signum, _frame):
            raise SystemExit(128 + signum)

        previous_term = signal.signal(signal.SIGTERM, stop_on_term)
        try:
            while True:
                index = claim_next(queue_path, gpu)
                if index is None:
                    break
                if index < 0 or index >= len(ordered):
                    raise ValueError(f'dataset_index {index} is outside 0..{len(ordered) - 1}')
                episode = ordered[index]
                self._env.current_episode = episode
                observations = self._env.reset()
                current = self._env.current_episode
                if (str(current.episode_id), current.scene_id) != (str(episode.episode_id), episode.scene_id):
                    raise RuntimeError(
                        f'reset moved off dataset_index {index}: '
                        f'{current.episode_id} {current.scene_id}'
                    )
                episode_id = current.episode_id
                scene_id = current.scene_id
                goal_object_id = getattr(current, 'goal_object_id', None)
                note_episode(
                    queue_path, gpu, index,
                    episode_id=episode_id, scene_id=scene_id, goal_object_id=goal_object_id,
                )
                print(
                    f'[Schedule] event=claim gpu={gpu} dataset_index={index} '
                    f'episode_id={episode_id} scene_id={scene_id} goal_object_id={goal_object_id}',
                    flush=True,
                )
                agent.reset()
                atomic_actions = bool(getattr(agent.config.ENVIRONMENT, 'ATOMIC_ACTIONS', False))
                from utils.atomic_actions import expand_atomic_actions
                tracker = _GoalVisibilityTracker(self._env, goal_object_id,
                                                 gpu=gpu, dataset_index=index)
                last_policy_action = None
                try:
                    while not self._env.episode_over:
                        step_started = time.perf_counter()
                        action = agent.act(observations)
                        executed_actions = 0
                        for primitive in expand_atomic_actions(action, atomic_actions):
                            observations = self._env.step(primitive)
                            executed_actions += 1
                            if self._env.episode_over:
                                break
                        episode_steps = self._env._elapsed_steps if atomic_actions else agent.total_steps
                        current_metrics = self._env.get_metrics()
                        agent.update_metrics(current_metrics)
                        tracker.update(observations)
                        _log_policy_step(env=self._env, agent=agent, action=action,
                                         executed=executed_actions, metrics=current_metrics,
                                         tracker=tracker, gpu=gpu, dataset_index=index)
                        last_policy_action = action.get('action') if isinstance(action, dict) else action
                        measured_step_seconds += time.perf_counter() - step_started
                        measured_steps += executed_actions
                        if episode_steps % 25 == 0 and not self._env.episode_over:
                            mean_step = measured_step_seconds / measured_steps
                            heartbeat(
                                queue_path, gpu,
                                dataset_index=index,
                                current_steps=episode_steps,
                                measured_steps=measured_steps,
                                measured_step_seconds=measured_step_seconds,
                            )
                            print(
                                f'[Progress] gpu={gpu} dataset_index={index} '
                                f'episode_steps={episode_steps} seconds_per_step={mean_step:.3f}',
                                flush=True,
                            )
                    raw = self._env.get_metrics()
                    episode_steps = self._env._elapsed_steps if atomic_actions else agent.total_steps
                    stored = dict(raw)
                    stored['episode_id'] = episode_id
                    stored['scene_id'] = scene_id
                    stored['goal_object_id'] = goal_object_id
                    stored['goal'] = (self._env.current_episode.object_category
                                      if getattr(agent, 'dataset', None) == 'coin' else agent.obj_goal)
                    stored['num_steps'] = episode_steps
                    stored['dataset_index'] = index
                    if atomic_actions:
                        stored['num_policy_steps'] = agent.total_steps
                        stored['step_unit'] = 'atomic_action'
                    _log_episode_end(env=self._env, agent=agent, raw=stored,
                                     tracker=tracker,
                                     termination=_termination_reason(self._env, last_policy_action),
                                     num_steps=episode_steps, gpu=gpu, dataset_index=index)
                    self._upsert_result(all_metrics, stored)
                    self._write_result_rows(results_file, all_metrics)
                    finish_episode(
                        queue_path, gpu, index,
                        measured_steps=measured_steps,
                        measured_step_seconds=measured_step_seconds,
                        success=raw['success'],
                        spl=raw['spl'],
                        softspl=raw['softspl'],
                        distance_to_goal=raw['distance_to_goal'],
                        num_steps=episode_steps,
                        goal=stored['goal'],
                    )
                    print(
                        f'[Schedule] event=done gpu={gpu} dataset_index={index} '
                        f'episode_id={episode_id} success={float(raw["success"])} '
                        f'spl={float(raw["spl"])} steps={episode_steps}',
                        flush=True,
                    )
                    print(f'Episode {index}: {raw}', flush=True)
                    _print_api_timing(agent, index, episode_id, gpu=gpu)
                except ModelRequestError as error:
                    if not error.retry_exhausted:
                        raise
                    defer_episode(
                        queue_path, gpu, index,
                        measured_steps=measured_steps,
                        measured_step_seconds=measured_step_seconds,
                        reason=str(error),
                    )
                    print(
                        f'[Schedule] event=deferred gpu={gpu} dataset_index={index} '
                        f'episode_id={episode_id} reason={error}',
                        flush=True,
                    )
                    continue
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            try:
                tasks_path = os.environ.get('PSGNAV_TASKS_FILE', '').strip()
                if not tasks_path:
                    log_path = os.environ.get('PSGNAV_LOG_FILE', '').strip()
                    if log_path.endswith('.log'):
                        tasks_path = log_path[:-4] + '.tasks.json'
                if tasks_path:
                    summary = write_tasks_file(queue_path, gpu, tasks_path)
                else:
                    summary = write_tasks_file(queue_path, gpu, progress_dir / 'tested_tasks.json')
                print(summary_line(summary), flush=True)
            except Exception as exc:
                print(f'[Schedule] event=summary gpu={gpu} error={type(exc).__name__}: {exc}', flush=True)
        if not all_metrics:
            return {}
        totals = defaultdict(float)
        for row in all_metrics:
            for name in ('distance_to_goal', 'success', 'spl', 'softspl'):
                totals[name] += float(row[name])
        return {name: value / len(all_metrics) for name, value in totals.items()}

    @staticmethod
    def _upsert_result(rows, stored):
        index = int(stored['dataset_index'])
        for position, row in enumerate(rows):
            if int(row['dataset_index']) == index:
                rows[position] = stored
                return
        rows.append(stored)

    @staticmethod
    def _write_result_rows(path, rows):
        path = Path(path)
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        with open(temporary, 'w', encoding='utf-8') as handle:
            for row in rows:
                handle.write('%s\n' % row)
        os.replace(temporary, path)

    def evaluate(
        self, agent: "Agent", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        if self._eval_remote is True:
            return self.remote_evaluate(agent, num_episodes)
        else:
            return self.local_evaluate(agent, num_episodes)
