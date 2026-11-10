"""CoIN detection vocabulary and merge radius, separate from InstanceNav's.

CoIN builds its GLIP prompt the way InstanceNav does: every goal category of the
evaluated split, then COIN_EXTRA_OBJECTS. InstanceNav, HSSD and HM3D-OVON keep
EXTRA_OBJECTS, their category lists and the 0.8 m merge radius.
"""
import collections
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from configs import dataset_categories as dc
from configs.dataset_registry import get_goal_merge_threshold

COIN_ROOT = Path(__file__).resolve().parents[1] / 'data' / 'datasets' / 'CoIN' / 'CoIN-Bench'
SPLITS = ('val_unseen', 'val_seen', 'val_seen_synonyms')
EXPECTED = {
    'val_unseen': ['mirror', 'picture', 'dresser', 'radiator', 'window glass', 'plant',
                   'hanging clothes', 'book', 'rug'],
    'val_seen': ['cabinet', 'bed', 'table', 'clothes', 'kitchen lower cabinet', 'cloth', 'blanket',
                 'wardrobe', 'desk', 'tv', 'bathroom cabinet', 'rack', 'display cabinet', 'chair',
                 'heater', 'board'],
    'val_seen_synonyms': ['kitchen cabinet', 'armchair', 'shelf'],
}
COIN_EXTRAS = ['curtain', 'oven', 'microwave', 'fireplace', 'piano']
# InstanceNav values, unchanged since before CoIN support (tests/goldens).
INSTANCENAV = ['chair', 'bed', 'plant', 'toilet', 'tv_monitor', 'sofa']
EXTRAS = ['heater', 'window', 'treadmill', 'exercise machine', 'staircase']


@pytest.fixture
def glip():
    """utils_glip keeps module globals; restore them for the other tests."""
    from utils import utils_glip
    names = ('categories_21_origin', 'categories_21', 'object_captions', 'projection',
             'projection_reverse', 'current_dataset')
    saved = {name: getattr(utils_glip, name) for name in names}
    yield utils_glip
    for name, value in saved.items():
        setattr(utils_glip, name, value)


def _words(phrases):
    return {word for phrase in phrases for word in phrase.split()}


@pytest.mark.parametrize('split', SPLITS)
def test_coin_categories_are_the_split_goals(split):
    expected = EXPECTED[split]
    assert dc.COIN_SPLIT_CATEGORIES[split] == expected
    assert dc.get_categories('coin', split=split) == expected
    for goal in expected:
        assert dc.get_categories('coin', goal, split=split) == expected
    assert dc.get_detection_categories('coin', expected[0], split=split) == expected + COIN_EXTRAS
    assert dc.get_category_count('coin', expected[0], split=split) == len(expected)
    assert dc.get_extra_objects('coin') == COIN_EXTRAS
    returned = dc.get_categories('coin', split=split)
    returned.append('x')
    assert dc.COIN_SPLIT_CATEGORIES[split] == expected  # a copy, not the table


@pytest.mark.parametrize('split', SPLITS)
def test_coin_glip_prompt(glip, split):
    expected = EXPECTED[split] + COIN_EXTRAS
    for goal in (EXPECTED[split][0], EXPECTED[split][-1]):
        glip.set_episode_goal_category(goal, 'coin', split=split)
        assert glip.categories_21_origin == EXPECTED[split]
        assert glip.categories_21 == expected
        assert glip.object_captions == '. '.join(expected) + '.'
        assert glip.projection == dict(enumerate(EXPECTED[split]))
    glip.init_categories('coin', split=split)
    assert glip.categories_21 == expected


def test_coin_rejects_unknown_split_and_foreign_goal(glip):
    for kwargs in ({}, {'split': 'val'}, {'split': 'train'}):
        with pytest.raises(ValueError):
            dc.get_categories('coin', 'mirror', **kwargs)
        with pytest.raises(ValueError):
            glip.set_episode_goal_category('mirror', 'coin', **kwargs)
    with pytest.raises(ValueError):
        dc.get_categories('coin', 'mirror', split='val_seen')  # a val_unseen goal
    with pytest.raises(ValueError):
        dc.get_categories('coin', 'window', split='val_unseen')
    with pytest.raises(ValueError):
        dc.get_category_count('coin', 'cabinet', split='val_unseen')
    with pytest.raises(ValueError):
        glip.init_categories('coin', goal_category='cabinet', split='val_unseen')


def test_other_datasets_unchanged(glip):
    assert dc.EXTRA_OBJECTS == EXTRAS
    for dataset in ('instancenav', 'hssd'):
        for kwargs in ({}, {'split': 'val'}, {'split': 'val_unseen'}):
            assert dc.get_categories(dataset, **kwargs) == INSTANCENAV
            assert dc.get_categories(dataset, 'chair', **kwargs) == INSTANCENAV
            assert dc.get_detection_categories(dataset, **kwargs) == INSTANCENAV + EXTRAS
            assert dc.get_category_count(dataset, **kwargs) == 6
        assert dc.get_extra_objects(dataset) == EXTRAS
        glip.init_categories(dataset)
        assert glip.categories_21_origin == INSTANCENAV and glip.categories_21 == INSTANCENAV + EXTRAS
        assert glip.object_captions == ('chair. bed. plant. toilet. tv_monitor. sofa. '
                                        'heater. window. treadmill. exercise machine. staircase.')
        before = glip.categories_21
        glip.set_episode_goal_category('chair', dataset, split='val')  # closed vocabulary: no rebuild
        assert glip.categories_21 is before
    for kwargs in ({}, {'split': 'val_unseen'}):
        assert dc.get_categories('hm3d_ovon', **kwargs) == []
        assert dc.get_categories('hm3d_ovon', 'cabinet', **kwargs) == ['cabinet']
        assert dc.get_detection_categories('hm3d_ovon', 'cabinet', **kwargs) == ['cabinet'] + EXTRAS
        assert dc.get_category_count('hm3d_ovon', 'cabinet', **kwargs) == 1
    assert dc.get_extra_objects('hm3d_ovon') == EXTRAS
    glip.set_episode_goal_category('cabinet', 'hm3d_ovon')
    assert glip.categories_21 == ['cabinet'] + EXTRAS
    returned = dc.get_extra_objects('instancenav')
    returned.append('x')
    assert dc.EXTRA_OBJECTS == EXTRAS


def test_coin_keeps_its_other_category_helpers():
    assert dc.get_threshold('mirror', 'coin') == dc.COIN_DETECTION_OVERRIDES.get('mirror', dc.COIN_MIN_DETECTIONS)
    assert dc.get_small_objects('coin') == ['plant', 'tv_monitor', 'book']
    assert dc.dataset_to_unified_name('window glass', 'coin') == 'window'
    assert dc.unified_to_dataset_name('window glass', 'coin') == 'window glass'
    assert dc.uses_episode_goal_categories('coin')


@pytest.mark.parametrize('raw,canonical', list(dc.COIN_TO_UNIFIED.items()))
def test_coin_mapping_preserves_raw_vocabulary(glip, raw, canonical):
    assert dc.dataset_to_unified_name(raw, 'coin') == canonical
    assert dc.goal_label_matches(raw, canonical, 'coin')
    assert dc.unified_to_dataset_name(canonical, 'coin', original_category=raw) == raw
    for split, words in EXPECTED.items():
        if raw in words:
            glip.set_episode_goal_category(raw, 'coin', split=split)
            assert glip.projection_reverse[raw] == words.index(raw)
            assert glip.categories_21 == words + COIN_EXTRAS


def test_coin_does_not_guess_ambiguous_reverse_or_merge_cabinets():
    assert dc.unified_to_dataset_name('chair', 'coin') == 'chair'
    assert dc.unified_to_dataset_name('chair', 'coin', original_category='armchair') == 'armchair'
    assert not dc.goal_label_matches('cabinet', 'bathroom cabinet', 'coin')
    assert not dc.goal_label_matches('cloth', 'clothes', 'coin')
    with pytest.raises(ValueError):
        dc.unified_to_dataset_name('chair', 'coin', original_category='bed')


def test_coin_node_gate_never_inherits_previous_goal():
    from graph.meta_func.io_func import IOFunc
    from omegaconf import DictConfig
    graph = IOFunc()
    graph.dataset, graph.node_space = 'coin', 'picture.'
    graph.threshold_list = {'picture': 4, 'chair': 5}
    graph.cfg = DictConfig({'obj_min_detections': 5})
    for goal in ['picture', 'mirror', 'book', 'chair', 'tv_monitor', 'window']:
        graph.set_obj_goal(goal, goal)
        assert graph.cfg.obj_min_detections == dc.get_threshold(goal, 'coin')
    graph.dataset = 'instancenav'
    graph.cfg = DictConfig({'obj_min_detections': 3})
    graph.set_obj_goal('chair', 'chair')
    assert graph.cfg.obj_min_detections == 5


def test_extent_exception_only_accepts_coin_books():
    from omegaconf import DictConfig
    from utils.utils_sg_uq.merge_utils import filter_objects
    import numpy as np
    small = np.array([[0., 0, 0], [.2, .1, .03]])
    def obj(caption, count=3):
        return {'pcd': SimpleNamespace(points=small), 'num_detections': count,
                'detection_captions': [caption] * count}
    cfg = DictConfig({'obj_min_points': 0, 'obj_min_detections': 3})
    assert len(filter_objects(cfg, [obj('book')])) == 0
    cfg.extent_filter_exemptions = ['book']
    kept = filter_objects(cfg, [obj('book'), obj('plant'), obj('book', 2)])
    assert len(kept) == 1 and kept[0]['detection_captions'] == ['book'] * 3


@pytest.mark.parametrize('raw,expected', [('plant', 'scenegraph'), ('tv', 'scenegraph'),
                                       ('book', 'scenegraph'), ('picture', 'navigation'),
                                       ('mirror', 'navigation'), ('dresser', 'navigation')])
def test_coin_goal_branch_still_runs_base_detector(raw, expected, monkeypatch):
    from utils.object_detection import ObjectDetectionManager
    import utils.object_detection as module
    monkeypatch.setattr(module, 'DETECTOR_TYPE', 'glip')
    manager = ObjectDetectionManager.__new__(ObjectDetectionManager)
    manager.agent = SimpleNamespace(scenegraph=SimpleNamespace(
        obj_goal=dc.dataset_to_unified_name(raw, 'coin'), small_objects=dc.get_small_objects('coin')))
    seen = []
    manager.collect_only_enabled = lambda: False
    manager._detect_with_glip = lambda obs: seen.append('base')
    manager._process_all_object_detections = lambda obs: seen.append('all_categories')
    manager._detect_goal_with_scenegraph = lambda obs: seen.append('scenegraph')
    manager._detect_goal_with_navigation_detector = lambda obs: seen.append('navigation')
    manager.detect_objects({})
    assert seen == ['base', 'all_categories', expected]


def _coin_category_names():
    """CoIN's 69 category names (index files when present, else the split lists)."""
    names = {name for categories in dc.COIN_SPLIT_CATEGORIES.values() for name in categories}
    indexes = [COIN_ROOT / split / f'{split}.json.gz' for split in SPLITS]
    for path in indexes:
        if path.is_file():
            with gzip.open(path, 'rt') as stream:
                names |= set(json.load(stream)['category_to_task_category_id'])
    if all(path.is_file() for path in indexes):
        assert len(names) == 69
    return names


def test_coin_extra_words_share_no_word_with_categories_rooms_or_doors():
    names = _coin_category_names()
    extra = _words(dc.COIN_EXTRA_OBJECTS)
    assert not extra & _words(names)
    assert not extra & _words(dc.ROOM_CATEGORIES)
    assert not extra & _words(dc.DOOR_CATEGORIES)
    # Why EXTRA_OBJECTS is not reused: 'window' is in 'window glass', 'heater' is a goal.
    assert {'window', 'heater'} <= _words(names)


def test_val_unseen_categories_share_no_word():
    counts = collections.Counter(word for name in dc.COIN_SPLIT_CATEGORIES['val_unseen']
                                 for word in set(name.split()))
    assert max(counts.values()) == 1, [word for word, count in counts.items() if count > 1]


@pytest.mark.parametrize('split', SPLITS)
def test_split_categories_match_the_dataset(split):
    shards = sorted((COIN_ROOT / split / 'content').glob('*.json.gz'))
    if not shards:
        pytest.skip(f'CoIN-Bench {split} not downloaded')
    counts = collections.Counter()
    for shard in shards:
        with gzip.open(shard, 'rt') as stream:
            counts.update(episode['object_category'] for episode in json.load(stream)['episodes'])
    categories = dc.COIN_SPLIT_CATEGORIES[split]
    assert len(categories) == len(set(categories)) and set(categories) == set(counts)
    ordered = [counts[name] for name in categories]
    assert ordered == sorted(ordered, reverse=True), dict(zip(categories, ordered))


def test_goal_merge_threshold():
    assert get_goal_merge_threshold('coin') == 0.25
    for dataset in ('instancenav', 'hssd', 'hm3d_ovon'):
        assert get_goal_merge_threshold(dataset) == 0.8


def test_preflight_vocabulary_problems():
    from tools.preflight_coin import vocabulary_problems
    episodes = [SimpleNamespace(object_category=name) for name in ('mirror', 'rug', 'mirror')]
    assert vocabulary_problems('val_unseen', episodes) == []
    problems = vocabulary_problems('val_unseen', episodes + [SimpleNamespace(object_category='cabinet')])
    assert len(problems) == 1 and "'cabinet'" in problems[0]
    problems = vocabulary_problems('val', episodes)
    assert len(problems) == 1 and 'COIN_SPLIT_CATEGORIES' in problems[0]
