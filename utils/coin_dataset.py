"""CoIN-Bench adapter for the bundled Habitat-Lab API (Context-Nav protocol).

CoIN-Bench ships an empty index ({split}.json.gz: category maps only) plus one
content shard per scene. Each episode names one target instance
(object_instance_id, e.g. "board_1451") and its full natural-language
instruction; each goal carries the target's view points (the success area) and
the positions of same-category distractors.

The policy receives the category and the full instruction, stored as
goal_attributes['intrinsic_attributes'] so the InstanceNav verifier reads it
unchanged. instruction_input ("Find the board"), camera_spec (an oracle pose),
the goal and its distractors stay evaluator-side.
"""
import glob
import gzip
import json
import os
import re
from typing import List, Optional

import attr
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.core.simulator import AgentState
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation

SCENE_DATASET_CONFIG = 'hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json'
DEFAULT_SCENES_DIR = 'data/scene_datasets'
# "<category>_<HM3D semantic id>"; categories may contain spaces ("window glass_12").
_INSTANCE_ID = re.compile(r'^(?P<category>.+)_(?P<semantic_id>\d+)$')


@attr.s(auto_attribs=True, kw_only=True)
class CoINGoal(ObjectGoal):
    distractors: List[List[float]] = attr.ib(factory=list)


@attr.s(auto_attribs=True, kw_only=True)
class CoINEpisode(NavigationEpisode):
    object_category: Optional[str] = None
    object_instance_id: Optional[str] = None
    goal_object_id: Optional[str] = None
    goal_attributes: Optional[dict] = None

    @property
    def goals_key(self):
        return os.path.basename(self.scene_id) + '_' + str(self.object_instance_id)


def scene_path(scene_id, scenes_dir):
    """CoIN 'hm3d/val/<scene>/<scene>.basis.glb' -> this checkout's hm3d_v0.2 tree."""
    relative = scene_id.replace('\\', '/').removeprefix('data/scene_datasets/')
    if relative.startswith('hm3d/'):
        relative = 'hm3d_v0.2/' + relative[len('hm3d/'):]
    return os.path.join(scenes_dir or DEFAULT_SCENES_DIR, relative)


def semantic_id(object_instance_id):
    """HM3D semantic id (string) encoded in a CoIN object_instance_id."""
    match = _INSTANCE_ID.match(str(object_instance_id))
    if match is None:
        raise ValueError(f'Unexpected CoIN object_instance_id {object_instance_id!r}')
    return match.group('semantic_id')


def episode_key(episode):
    """(scene basename, episode id): unique within a CoIN split."""
    return os.path.basename(episode.scene_id), str(episode.episode_id)


def _view_point(raw):
    state = raw['agent_state']
    return ObjectViewLocation(
        agent_state=AgentState(position=state['position'], rotation=state['rotation']),
        iou=raw.get('iou'))


def _goal(raw):
    return CoINGoal(
        position=raw['position'], radius=raw.get('radius'), object_id=str(raw['object_id']),
        object_name=raw.get('object_name'), object_name_id=raw.get('object_name_id'),
        object_category=raw.get('object_category'), room_id=raw.get('room_id'),
        room_name=raw.get('room_name'), view_points=[_view_point(v) for v in raw['view_points']],
        distractors=[[float(c) for c in d] for d in raw.get('distractors') or []])


@registry.register_dataset(name='CoINInstanceNav-v1')
class CoINDataset(Dataset):
    def __init__(self, config=None):
        self.episodes = []
        self.category_to_task_category_id = {}
        self.category_to_scene_annotation_category_id = {}
        if config is None:
            return
        index_path = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(index_path, 'rt') as stream:
            index = json.load(stream)
        if index.get('episodes'):
            raise ValueError(f'{index_path}: expected an empty CoIN index with per-scene content shards')
        self.category_to_task_category_id = index['category_to_task_category_id']
        self.category_to_scene_annotation_category_id = index.get('category_to_scene_annotation_category_id', {})
        shards = sorted(glob.glob(os.path.join(os.path.dirname(index_path), 'content', '*.json.gz')))
        if not shards:
            raise ValueError(f'No CoIN content shards next to {index_path}')
        for shard in shards:
            with gzip.open(shard, 'rt') as stream:
                self.from_json(stream.read(), config.SCENES_DIR)
        unknown = sorted({ep.object_category for ep in self.episodes} - set(self.category_to_task_category_id))
        if unknown:
            raise ValueError(f'CoIN categories missing from category_to_task_category_id: {unknown}')
        self.episodes.sort(key=lambda ep: (ep.scene_id, int(ep.episode_id)))
        keys = [episode_key(ep) for ep in self.episodes]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate (scene, episode_id) in CoIN split')
        self._select(config)
        if not self.episodes:
            raise ValueError('No CoIN episodes selected')

    def _select(self, config):
        manifest_path = getattr(config, 'SUBSET_MANIFEST', '')
        if manifest_path:
            with open(manifest_path) as stream:
                manifest = json.load(stream)
            if manifest.get('split') not in (None, config.SPLIT):
                raise ValueError(f'CoIN manifest {manifest_path} is for split {manifest["split"]}, not {config.SPLIT}')
            selected = {tuple(key) for key in manifest['episode_keys']}
            self.episodes = [ep for ep in self.episodes if episode_key(ep) in selected]
            if len(self.episodes) != len(selected) or len(selected) != manifest['selected_count']:
                raise ValueError('CoIN subset manifest does not match the loaded split')
            print(f'[CoIN] Fixed subset: {len(self.episodes)} episodes, seed={manifest.get("seed")}')
        scenes = sorted({ep.scene_id for ep in self.episodes})
        left, right = getattr(config, 'SPLIT_L', -1), getattr(config, 'SPLIT_R', -1)
        if left >= 0:
            chosen = set(scenes[left:None if right < 0 else right])
            self.episodes = [ep for ep in self.episodes if ep.scene_id in chosen]

    def from_json(self, json_str, scenes_dir=None):
        """Parse one CoIN content shard (goals keyed '<scene>.basis.glb_<instance>')."""
        data = json.loads(json_str)
        goals = data['goals']
        for raw in data['episodes']:
            key = os.path.basename(raw['scene_id']) + '_' + raw['object_instance_id']
            raw_goals = goals.get(key)
            if not raw_goals or len(raw_goals) != 1:
                raise ValueError(f'CoIN goal {key}: expected exactly one goal entry')
            if str(raw_goals[0]['object_id']) != raw['object_instance_id']:
                raise ValueError(f'CoIN goal {key}: object_id {raw_goals[0]["object_id"]!r} does not match')
            if raw.get('additional_obj_config_paths'):
                raise ValueError(f'CoIN episode {raw["episode_id"]}: additional objects are not supported')
            instruction = str(raw.get('instruction') or '').strip()
            if not instruction:
                raise ValueError(f'CoIN episode {raw["episode_id"]}: empty instruction')
            self.episodes.append(CoINEpisode(
                episode_id=str(raw['episode_id']),
                scene_id=scene_path(raw['scene_id'], scenes_dir),
                scene_dataset_config=os.path.join(scenes_dir or DEFAULT_SCENES_DIR, SCENE_DATASET_CONFIG),
                start_position=raw['start_position'],
                start_rotation=raw['start_rotation'],
                start_room=raw.get('start_room'),
                object_category=raw['object_category'],
                object_instance_id=raw['object_instance_id'],
                goal_object_id=semantic_id(raw['object_instance_id']),
                # Full instruction, no user interaction (Context-Nav). Extrinsic left
                # empty so the verifier sends the instruction exactly once.
                goal_attributes={'intrinsic_attributes': instruction, 'extrinsic_attributes': ''},
                goals=[_goal(raw_goals[0])],
            ))
