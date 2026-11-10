"""PSL InstanceNav adapter for the bundled Habitat-Lab API.

Metrics see only the designated object's viewpoints, never all category instances.
The policy receives the textual attributes; goal IDs/coordinates stay evaluator-side.
"""
import gzip
import json
import os
from typing import Any, Optional

import attr
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.core.simulator import AgentState
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation


@attr.s(auto_attribs=True, kw_only=True)
class InstanceEpisode(NavigationEpisode):
    object_category: Optional[str] = None
    goal_object_id: Optional[str] = None
    goal_image_id: Optional[Any] = None
    goal_attributes: Optional[dict] = None

    @property
    def goals_key(self):
        return os.path.basename(self.scene_id).split('.')[0] + '_' + str(self.goal_object_id)


@registry.register_dataset(name='PSGInstanceNav-v1')
class InstanceNavDataset(Dataset):
    def __init__(self, config=None):
        self.episodes = []
        self.category_to_task_category_id = {}
        if config is None:
            return
        with gzip.open(config.DATA_PATH.format(split=config.SPLIT), 'rt') as f:
            self.from_json(f.read(), config.SCENES_DIR)
        scenes = sorted({ep.scene_id for ep in self.episodes})
        left, right = getattr(config, 'SPLIT_L', -1), getattr(config, 'SPLIT_R', -1)
        if left >= 0:
            selected = set(scenes[left:None if right < 0 else right])
            self.episodes = [ep for ep in self.episodes if ep.scene_id in selected]
        if not self.episodes:
            raise ValueError('No InstanceNav episodes selected')

    def from_json(self, json_str, scenes_dir=None):
        data = json.loads(json_str)
        self.category_to_task_category_id = data['category_to_task_category_id']
        allowed_goal = {field.name for field in attr.fields(ObjectGoal)}
        for raw in data['episodes']:
            episode = InstanceEpisode(**raw)
            raw_goal = data['goals'][episode.goals_key]
            goal = ObjectGoal(**{k:v for k,v in raw_goal.items() if k in allowed_goal and k != 'view_points'})
            goal.view_points = [ObjectViewLocation(
                agent_state=AgentState(**view['agent_state']), iou=view.get('iou')
            ) for view in raw_goal['view_points']]
            episode.goals = [goal]
            if not episode.goal_attributes:
                raise ValueError(f'Missing description for {episode.goals_key}')
            if scenes_dir is not None:
                relative = episode.scene_id.removeprefix('data/scene_datasets/')
                episode.scene_id = os.path.join(scenes_dir, relative)
                episode.scene_dataset_config = os.path.join(
                    scenes_dir, 'hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json')
            self.episodes.append(episode)
