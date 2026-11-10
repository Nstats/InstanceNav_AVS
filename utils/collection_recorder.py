"""Passive, replayable evidence for collect-only runs. Never used by the policy.

Navigation tracks and these observation tracks deliberately have separate IDs.
All detector proposals are associated before their model answers are known.
No semantic annotations or target positions are read here.
"""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from utils.log_utils import log_event

SCHEMA_VERSION = 2


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def append_record(directory, tag, record):
    # Fail loudly on disk errors: a long collection with missing evidence is unusable.
    with (Path(directory) / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'tag': tag, **record}, ensure_ascii=False, default=_json,
                                allow_nan=False) + '\n')


def save_image(directory, image):
    image = image.convert('RGB')
    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    relative = f'images/{digest}.png'
    path = Path(directory) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        image.save(path, format='PNG')
    return {'path': relative, 'pixel_sha256': digest, 'size': list(image.size)}


def model_settings(client):
    # Explicit allowlist: never serialize the client, URL, headers or credentials.
    backend = getattr(client, 'backend', None)
    return {'backend': backend, 'model': getattr(client, 'model', None),
            'temperature': 0.0 if backend and backend != 'local' else None,
            'enable_thinking': False if backend == 'openai' else None,
            'max_tokens': getattr(client, 'max_tokens', None),
            'cache_size': getattr(client, 'cache_size', None),
            'retries': getattr(client, 'retries', None)}


def request_metadata(client):
    getter = getattr(client, 'last_request_metadata', None)
    return getter() if callable(getter) else {'cache_hit': None, 'cache_key': None}


class CollectionRecorder:
    def __init__(self, agent, episode):
        from utils.instance_evidence import PositiveEvidenceStore
        logfile = os.environ.get('PSGNAV_LOG_FILE')
        if not logfile:
            raise RuntimeError('Collect-only recording requires PSGNAV_LOG_FILE (set by the launcher)')
        self.directory = Path(logfile).resolve().with_suffix('.collect')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.agent = agent
        self.store = PositiveEvidenceStore(agent.goal_merge_threshold)
        self.identity = {'schema_version': SCHEMA_VERSION, 'episode_attempt_id': uuid4().hex,
                         'episode_id': str(getattr(episode, 'episode_id', '')),
                         'scene_id': str(getattr(episode, 'scene_id', ''))}
        self.last_frame = None
        self.finished = False
        self.emit('Collect Episode', {
            'category': (getattr(agent, 'dataset_goal_category', agent.obj_goal)
                         if getattr(agent, 'dataset', None) == 'coin' else agent.obj_goal),
            **({'canonical_category': agent.obj_goal} if getattr(agent, 'dataset', None) == 'coin' else {}),
            'description': {k: agent.goal_attributes.get(k, '') for k in
                            ('intrinsic_attributes', 'extrinsic_attributes')},
            'association_distance_m': agent.goal_merge_threshold,
            'confirmation_distance_m': agent.distance_threshold,
            'collection_policy': 'retire_at_intended_lock_then_explore',
            'association_method': 'node_id_then_spatial_linf_one_to_one',
            'track_ids_are_oracle': False,
        })

    def context(self):
        env = getattr(getattr(self.agent, 'simulator', None), '_env', None)
        return {**self.identity, 'episode': getattr(self.agent, 'count_episodes', None),
                'step': self.agent.total_steps,
                'atomic_steps_before': getattr(env, '_elapsed_steps', None)}

    def emit(self, tag, fields):
        record = {**self.context(), **fields}
        append_record(self.directory, tag, record)
        log_event(tag, **record)
        return record

    def begin_frame(self, observations, candidates, source, scores):
        frame_id = uuid4().hex
        rgb = save_image(self.directory, Image.fromarray(observations['rgb'][..., :3]))
        depth = observations.get('depth', getattr(self.agent, 'depth', None))
        depth_asset = None
        if depth is not None:
            depth = np.asarray(depth)
            digest = hashlib.sha256(str((depth.shape, str(depth.dtype))).encode() + depth.tobytes()).hexdigest()
            relative = f'depth/{digest}.npz'
            path = self.directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                np.savez_compressed(path, depth=depth)
            depth_asset = {'path': relative, 'array_sha256': digest,
                           'kind': 'policy_depth_rgb_aligned_for_coin', 'invalid_sentinel': 100}
        # Snapshot the previous state before associate() updates it in place.
        previous = {t.track_id: (t.step, None if t.gps is None else t.gps.copy())
                    for t in self.store.tracks}
        tracks = self.store.associate(candidates, self.agent.total_steps)
        records = []
        for i, (candidate, track) in enumerate(zip(candidates, tracks)):
            old = previous.get(track.track_id)
            distance = candidate['distance']
            records.append({
                'box_index': i, 'collect_track_id': track.track_id,
                'node_id': str(candidate['node_id']) if candidate['node_id'] is not None else None,
                'box': list(map(float, candidate['box'])),
                'gps': candidate['gps'],
                'distance_m': float(distance) if np.isfinite(distance) else None,
                'detector_score': scores[i], 'localized': candidate['gps'] is not None,
                'association': ('node_id' if candidate['node_id'] is not None else 'spatial') if old else
                               ('new' if candidate['gps'] is not None else 'new_unlocalized'),
                'previous_step': old[0] if old else None,
                'association_delta_m': float(np.linalg.norm(candidate['gps'] - old[1]))
                    if old and old[1] is not None and candidate['gps'] is not None else None,
            })
        pose = getattr(self.agent, 'full_pose', None)
        if hasattr(pose, 'detach'):
            pose = pose.detach().cpu().numpy()
        record = self.emit('Collect Observation', {
            'frame_id': frame_id, 'source': source, 'rgb': rgb, 'depth': depth_asset,
            'gps_sensor': observations.get('gps'), 'compass_sensor': observations.get('compass'),
            'mapping_view_angles_deg': getattr(getattr(self.agent, 'sem_map_module', None), 'view_angles', None),
            'agent_pose': pose, 'candidates': records,
        })
        self.last_frame = record
        return {**self.context(), 'frame_id': frame_id,
                'collect_directory': str(self.directory), 'rgb_pixel_sha256': rgb['pixel_sha256']}

    def finish_frame(self, frame_context, tracks, kept, stages, retired_ids):
        self.emit('Collect Decisions', {
            'frame_id': frame_context['frame_id'], 'stages': stages,
            'candidates': [
                {'box_index': i, 'collect_track_id': c['collect_track_id'],
                 'navigation_track_id': tracks[i].track_id if tracks[i] is not None else None,
                 'kept': i in kept,
                 'retired': tracks[i] is not None and tracks[i].track_id in retired_ids,
                 'decisions': {s['stage']: s['matches'][s['box_indices'].index(i)]
                               for s in stages if i in s['box_indices']}}
                for i, c in enumerate(self.last_frame['candidates'])],
        })

    def intended_lock(self, track, path):
        # The current detection frame links navigation IDs back to passive IDs.
        if self.last_frame is None:
            raise RuntimeError('Intended lock without a recorded observation')
        self.emit('Collect Intended Lock', {
            'frame_id': self.last_frame['frame_id'], 'path': path,
            'navigation_track_id': track.track_id, 'box': track.current['box'],
            'goal_gps': track.gps, 'votes': len(track.votes),
        })

    def finish_episode(self):
        if self.finished:
            return
        env = self.agent.simulator._env
        self.emit('Collect End', {'atomic_steps': env._elapsed_steps,
                                 'budget': self.agent.max_episode_steps,
                                 'found_goal': bool(self.agent.found_goal)})
        self.finished = True
