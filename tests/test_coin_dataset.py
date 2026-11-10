"""CoIN-Bench adapter: shard parsing, scene paths, policy-visible fields, subsets."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import utils.coin_dataset as cd

COIN_ROOT = Path('data/datasets/CoIN/CoIN-Bench')
SCENE = 'hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb'


def _goal(instance='picture_407'):
    return {'position': [1.0, 0.1, 2.0], 'radius': None, 'object_id': instance, 'object_name': None,
            'object_name_id': None, 'object_category': instance.rsplit('_', 1)[0], 'room_id': None,
            'room_name': None, 'distractors': [[3.0, 0.1, 4.0], [5.0, 0.1, 6.0]],
            'view_points': [{'agent_state': {'position': [1.0, 0.1, 2.0], 'rotation': [0, 0, 0, 1]}, 'iou': 0.4}]}


def _episode(episode_id, instance='picture_407', **extra):
    raw = {'episode_id': episode_id, 'scene_id': SCENE, 'scene_dataset_config': 'default',
           'additional_obj_config_paths': [], 'start_position': [0.0, 0.1, 0.0],
           'start_rotation': [0, 0.7071, 0, 0.7071], 'info': None, 'start_room': None, 'shortest_paths': None,
           'goals': [], 'object_category': instance.rsplit('_', 1)[0], 'object_instance_id': instance,
           'instruction': 'the picture of sailboats above the sofa.', 'instruction_input': 'Find the picture',
           'camera_spec': {'oracle': True}}
    raw.update(extra)
    return raw


def _shard(episodes, goals=None):
    goals = goals if goals is not None else {'TEEsavR23oF.basis.glb_picture_407': [_goal()]}
    return json.dumps({'goals': goals, 'goals_by_instance': {}, 'episodes': episodes})


def test_scene_path_and_semantic_id():
    assert cd.scene_path(SCENE, 'data/scene_datasets') == \
        'data/scene_datasets/hm3d_v0.2/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb'
    assert cd.scene_path('data/scene_datasets/' + SCENE, None).startswith('data/scene_datasets/hm3d_v0.2/')
    assert cd.semantic_id('window glass_12') == '12'
    with pytest.raises(ValueError):
        cd.semantic_id('picture')


def test_shard_episode_fields():
    dataset = cd.CoINDataset()
    dataset.from_json(_shard([_episode(7)]), 'data/scene_datasets')
    (episode,) = dataset.episodes
    assert episode.episode_id == '7' and episode.goal_object_id == '407'
    assert episode.scene_dataset_config.endswith('hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json')
    # Full instruction, once; instruction_input and the oracle camera_spec are not stored.
    assert episode.goal_attributes == {'intrinsic_attributes': 'the picture of sailboats above the sofa.',
                                       'extrinsic_attributes': ''}
    assert not hasattr(episode, 'instruction_input') and not hasattr(episode, 'camera_spec')
    goal = episode.goals[0]
    assert goal.object_id == 'picture_407' and len(goal.view_points) == 1
    assert goal.view_points[0].agent_state.position == [1.0, 0.1, 2.0]
    assert goal.distractors == [[3.0, 0.1, 4.0], [5.0, 0.1, 6.0]]
    assert episode.goals_key == 'TEEsavR23oF.basis.glb_picture_407'  # gitleaks:allow (non-secret package version or test fixture ID)


@pytest.mark.parametrize('episode, goals, message', [
    (_episode(1, instruction='  '), None, 'empty instruction'),
    (_episode(1, additional_obj_config_paths=['x']), None, 'additional objects'),
    (_episode(1, instance='picture_9'), None, 'expected exactly one goal'),
    (_episode(1), {'TEEsavR23oF.basis.glb_picture_407': [_goal(), _goal()]}, 'expected exactly one goal'),
    (_episode(1), {'TEEsavR23oF.basis.glb_picture_407': [_goal('picture_408')]}, 'does not match'),
])
def test_shard_rejects_malformed_episodes(episode, goals, message):
    with pytest.raises(ValueError, match=message):
        cd.CoINDataset().from_json(_shard([episode], goals), 'data/scene_datasets')


def _write_split(root, split, episodes):
    content = root / split / 'content'
    content.mkdir(parents=True)
    with gzip.open(root / split / f'{split}.json.gz', 'wt') as stream:
        json.dump({'episodes': [], 'category_to_task_category_id': {'picture': 0},
                   'category_to_scene_annotation_category_id': {'picture': 0}}, stream)
    with gzip.open(content / 'TEEsavR23oF.json.gz', 'wt') as stream:
        stream.write(_shard(episodes))


def _config(root, split='val_seen', **extra):
    return SimpleNamespace(DATA_PATH=str(root / '{split}' / '{split}.json.gz'), SPLIT=split,
                           SCENES_DIR='data/scene_datasets', **extra)


def test_dataset_sorts_and_applies_manifest(tmp_path):
    _write_split(tmp_path, 'val_seen', [_episode(12), _episode(3), _episode(7)])
    dataset = cd.CoINDataset(_config(tmp_path))
    assert [ep.episode_id for ep in dataset.episodes] == ['3', '7', '12']
    manifest = tmp_path / 'subset.json'
    manifest.write_text(json.dumps({'split': 'val_seen', 'seed': 1, 'selected_count': 2,
                                    'episode_keys': [['TEEsavR23oF.basis.glb', '12'], ['TEEsavR23oF.basis.glb', '3']]}))
    subset = cd.CoINDataset(_config(tmp_path, SUBSET_MANIFEST=str(manifest)))
    assert [ep.episode_id for ep in subset.episodes] == ['3', '12']
    with pytest.raises(ValueError, match='not val_unseen'):
        _write_split(tmp_path, 'val_unseen', [_episode(3)])
        cd.CoINDataset(_config(tmp_path, split='val_unseen', SUBSET_MANIFEST=str(manifest)))
    manifest.write_text(json.dumps({'split': 'val_seen', 'selected_count': 1,
                                    'episode_keys': [['TEEsavR23oF.basis.glb', '99']]}))
    with pytest.raises(ValueError, match='does not match'):
        cd.CoINDataset(_config(tmp_path, SUBSET_MANIFEST=str(manifest)))


def test_dataset_rejects_duplicates_and_unknown_categories(tmp_path):
    _write_split(tmp_path, 'val_seen', [_episode(3), _episode(3)])
    with pytest.raises(ValueError, match='Duplicate'):
        cd.CoINDataset(_config(tmp_path))
    other = tmp_path / 'other'
    _write_split(other, 'val_seen', [_episode(3, instance='vase_5')])
    with gzip.open(other / 'val_seen' / 'content' / 'TEEsavR23oF.json.gz', 'wt') as stream:
        stream.write(_shard([_episode(3, instance='vase_5')], {'TEEsavR23oF.basis.glb_vase_5': [_goal('vase_5')]}))
    with pytest.raises(ValueError, match='missing from category_to_task_category_id'):
        cd.CoINDataset(_config(other))


def test_full_instruction_reaches_verifier_and_exploration_prompts(monkeypatch):
    import numpy as np
    from utils.instance_verifier import verify_candidates
    from utils.landmark_selector import LandmarkSelector
    dataset = cd.CoINDataset()
    dataset.from_json(_shard([_episode(7)]), 'data/scene_datasets')
    attributes = dataset.episodes[0].goal_attributes
    prompts = []

    def complete(prompt, image, **options):
        prompts.append(prompt)
        answer = '{"matches": ["yes"]}'
        options['validator'](answer)
        return answer

    kept = verify_candidates(SimpleNamespace(complete=complete), np.zeros((40, 40, 3), np.uint8),
                             [[1, 1, 20, 20]], 'picture', attributes)
    assert kept == [0]
    instruction = 'the picture of sailboats above the sofa.'
    assert prompts[0].count(instruction) == 1
    assert 'Find the picture' not in prompts[0] and 'oracle' not in prompts[0]
    monkeypatch.setenv('ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION', 'true')
    selector = LandmarkSelector.__new__(LandmarkSelector)
    selector.agent = SimpleNamespace(goal_attributes=attributes)
    assert selector._with_instance_description('Pick a landmark.').count(instruction) == 1
    monkeypatch.setenv('ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION', 'false')
    assert selector._with_instance_description('Pick a landmark.') == 'Pick a landmark.'


@pytest.mark.skipif(not COIN_ROOT.is_dir(), reason='CoIN-Bench not downloaded')
@pytest.mark.parametrize('split, episodes, targets', [
    ('val_seen', 831, 86), ('val_seen_synonyms', 359, 36), ('val_unseen', 459, 47)])
def test_real_splits(split, episodes, targets):
    dataset = cd.CoINDataset(_config(COIN_ROOT, split=split))
    assert len(dataset.episodes) == episodes
    assert len({(ep.scene_id, ep.object_instance_id) for ep in dataset.episodes}) == targets
    assert all(Path(ep.scene_id).is_file() for ep in dataset.episodes[:: max(1, episodes // 20)])
    assert all(ep.goals[0].distractors and ep.goals[0].view_points for ep in dataset.episodes)
