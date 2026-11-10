"""HM3D-OVON dataset adapter for the bundled Habitat-Lab version.

The released OVON manifests use ``hm3d/val/...`` scene references while this
checkout stores the same scenes under ``hm3d_v0.2/val/...``.  The base Habitat
ObjectNav loader already handles OVON's empty top-level manifest and content
shards; this adapter normalizes those scene references and keeps the episode
metadata needed by the policy.
"""
import json
import os

from habitat.core.registry import registry
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1


def _normalize_scene_path(scene_id: str) -> str:
    scene_id = scene_id.replace("\\", "/")
    if scene_id.startswith("data/scene_datasets/"):
        scene_id = scene_id[len("data/scene_datasets/"):]
    if scene_id.startswith("hm3d/val//"):
        scene_id = scene_id[len("hm3d/val//"):]
        scene_id = "hm3d_v0.2/val/" + scene_id
    elif scene_id.startswith("hm3d/val/"):
        scene_id = scene_id[len("hm3d/"):]
        scene_id = "hm3d_v0.2/" + scene_id
    return scene_id.replace("/val//", "/val/")


@registry.register_dataset(name="OVON-v1")
class OVONDatasetV1(ObjectNavDatasetV1):
    """ObjectNav loader with HM3D-OVON scene path compatibility."""

    def __init__(self, config=None):
        super().__init__(config)
        # The public OVON manifest has one validation scene that is not part
        # of the locally available HM3D asset subset.  Keep the run alive and
        # report the omission instead of failing when Habitat opens that scene.
        missing = [ep.scene_id for ep in self.episodes if not os.path.exists(ep.scene_id)]
        if missing:
            missing_scenes = sorted(set(missing))
            print(
                f"[OVON] Skipping {len(missing_scenes)} unavailable local scene(s): "
                + ", ".join(os.path.basename(os.path.dirname(path)) for path in missing_scenes)
            )
            self.episodes = [ep for ep in self.episodes if os.path.exists(ep.scene_id)]
        manifest_path = getattr(config, 'SUBSET_MANIFEST', '') if config else ''
        if manifest_path:
            with open(manifest_path) as stream:
                manifest = json.load(stream)
            selected = {tuple(key) for key in manifest['episode_keys']}
            self.episodes = [ep for ep in self.episodes
                             if (os.path.basename(ep.scene_id), str(ep.episode_id)) in selected]
            if len(self.episodes) != len(selected) or len(selected) != manifest['selected_count']:
                raise ValueError('OVON subset manifest does not match the loaded dataset')
            print(f'[OVON] Fixed subset: {len(self.episodes)} episodes, seed={manifest["seed"]}')

    def from_json(self, json_str: str, scenes_dir=None) -> None:
        data = json.loads(json_str)
        for episode in data.get("episodes", []):
            episode["scene_id"] = _normalize_scene_path(episode["scene_id"])
            episode["scene_dataset_config"] = (
                os.path.join(scenes_dir or 'data/scene_datasets',
                             "hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json")
            )
        # ObjectNavDatasetV1 handles goal deserialization, content shards,
        # children_object_categories, and category mappings.
        super().from_json(json.dumps(data), scenes_dir=scenes_dir)
