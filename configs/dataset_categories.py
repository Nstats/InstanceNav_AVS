"""
Dataset-specific category configurations for ObjectNav benchmarks.

This module defines object categories, thresholds, and mappings for HSSD ObjectNav.
"""

from configs.dataset_registry import is_open_vocabulary_dataset

# Episode/split-driven vocabularies. CoIN keeps raw detector words separately
# from its canonical policy names; OVON remains identity-mapped.
# The category list is the episode's goal for hm3d_ovon and the split's goal
# categories (COIN_SPLIT_CATEGORIES) for coin.
OPEN_VOCAB_EPISODE_DATASETS = ('hm3d_ovon', 'coin')

HSSD_CATEGORIES = [
    'chair', 'bed', 'potted_plant', 'toilet', 'tv', 'couch'
]

HSSD_TO_UNIFIED = {
    'potted_plant': 'plant',
    'tv': 'tv_monitor',
    'couch': 'sofa',
    'tv_monitor': 'tv_monitor',
}

UNIFIED_TO_HSSD = {
    'plant': 'potted_plant',
    'tv_monitor': 'tv',
    'sofa': 'couch',
}

HSSD_SMALL_OBJECTS = ['potted_plant', 'tv']

HSSD_THRESHOLDS = {
    'chair': 5, 'bed': 5, 'potted_plant': 3, 'toilet': 2, 'tv': 2, 'couch': 4
}

EXTRA_OBJECTS = ['heater', 'window', 'treadmill', 'exercise machine', 'staircase']

# CoIN-Bench goal categories of each split: every episode's object_category, ordered by
# episode count (data/datasets/CoIN/CoIN-Bench/<split>/content/*.json.gz; counts in the
# comments). They play the role of the six InstanceNav categories: the base of the GLIP
# prompt. tests/test_coin_vocabulary.py recomputes them from the dataset.
COIN_SPLIT_CATEGORIES = {
    'val_unseen': [  # 459 episodes
        'mirror',           # 158
        'picture',          # 134
        'dresser',          # 50
        'radiator',         # 40
        'window glass',     # 20
        'plant',            # 20
        'hanging clothes',  # 17
        'book',             # 10
        'rug',              # 10
    ],
    'val_seen': [  # 831 episodes
        'cabinet',                # 302
        'bed',                    # 149
        'table',                  # 119
        'clothes',                # 40
        'kitchen lower cabinet',  # 40
        'cloth',                  # 30
        'blanket',                # 30
        'wardrobe',               # 20
        'desk',                   # 20
        'tv',                     # 20
        'bathroom cabinet',       # 20
        'rack',                   # 10
        'display cabinet',        # 10
        'chair',                  # 10
        'heater',                 # 10
        'board',                  # 1
    ],
    'val_seen_synonyms': [  # 359 episodes
        'kitchen cabinet',  # 249
        'armchair',         # 70
        'shelf',            # 40
    ],
}

# CoIN's counterpart of EXTRA_OBJECTS: words that only enlarge the GLIP prompt; they are
# never goals and never enter obj_locations. Rule: the most frequent indoor objects in
# CoIN instructions (169 distinct instructions over the three splits: curtain 20, oven 11,
# microwave 4, fireplace 3, piano 2) after removing
#   - words in any of CoIN's 69 category names (window, refrigerator, railing),
#   - near-synonyms of a CoIN category (painting/frame -> picture, countertop -> counter,
#     light -> lamp, fridge -> refrigerator, heater -> radiator),
#   - surfaces (wall, floor, ceiling) and words already in DOOR_CATEGORIES (hallway),
#   - stove (the same object as oven);
# the tie at count 2 is broken by val_unseen frequency (piano).
# EXTRA_OBJECTS does not carry over: the goal matcher (utils/object_detection.py,
# `label in goal.split()`) would take a 'window' box for the 'window glass' goal, and
# 'heater' is a val_seen goal and the same object as val_unseen's radiator.
COIN_EXTRA_OBJECTS = ['curtain', 'oven', 'microwave', 'fireplace', 'piano']

COIN_TO_UNIFIED = {
    'tv': 'tv_monitor',
    'dresser': 'chest_of_drawers',
    'window glass': 'window',
    'hanging clothes': 'clothes',
    'heater': 'radiator',
    'armchair': 'chair',
}
COIN_SMALL_OBJECTS = ('plant', 'tv_monitor', 'book')
# 20261109 offline replay: 47 target groups, five-fold F1 improvement over 3.
# Conditional proxy calibration; see docs/coin_calibration.md for limits.
COIN_MIN_DETECTIONS = 1
COIN_DETECTION_OVERRIDES = {}


def coin_policy_categories():
    """Serializable settings, shared by run manifests and runtime diagnostics."""
    return {'category_mapping': dict(COIN_TO_UNIFIED),
            'small_objects': list(COIN_SMALL_OBJECTS),
            'node_min_detections': COIN_MIN_DETECTIONS,
            'node_min_detections_overrides': dict(COIN_DETECTION_OVERRIDES),
            'extent_filter_exemptions': ['book']}


def goal_label_matches(label, goal, dataset='hssd'):
    """CoIN compares canonical phrases, without conflating cabinet subtypes.

    Other datasets retain the historical word-level matching rule.
    """
    label, goal = str(label).lower(), str(goal).lower()
    if dataset.lower() == 'coin':
        return dataset_to_unified_name(label.strip(), dataset) == dataset_to_unified_name(goal.strip(), dataset)
    return goal == label or goal in label.split() or label in goal.split()
ROOM_CATEGORIES = [
    'bedroom', 'living room', 'bathroom', 'kitchen', 'dining room',
    'office room', 'gym', 'lounge', 'laundry room'
]
DOOR_CATEGORIES = ['doorway', 'hallway']

def get_coin_categories(split, goal_category=None):
    """Goal categories of a CoIN split (a copy); raises for an unknown split or a goal outside it."""
    if split not in COIN_SPLIT_CATEGORIES:
        raise ValueError(f"CoIN split {split!r} has no category list; known splits: "
                         f"{sorted(COIN_SPLIT_CATEGORIES)}")
    categories = list(COIN_SPLIT_CATEGORIES[split])
    if goal_category is not None and goal_category not in categories:
        raise ValueError(f"CoIN goal {goal_category!r} is not a {split} category: {categories}")
    return categories


def get_categories(dataset='hssd', goal_category=None, split=None):
    """
    Get HSSD ObjectNav categories in unified internal naming.

    split is used by coin only (its categories are the split's goal categories).
    """
    dataset = dataset.lower()
    if dataset == 'coin':
        return get_coin_categories(split, goal_category=goal_category)
    if dataset in OPEN_VOCAB_EPISODE_DATASETS:
        # OVON categories are open vocabulary and change per episode.
        return [goal_category] if goal_category else []
    if dataset in ('hssd', 'instancenav'):
        return [HSSD_TO_UNIFIED.get(cat, cat) for cat in HSSD_CATEGORIES]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_extra_objects(dataset='hssd'):
    """Words that only enlarge the GLIP prompt (never goals): CoIN's own, else EXTRA_OBJECTS."""
    if dataset.lower() == 'coin':
        return list(COIN_EXTRA_OBJECTS)
    return list(EXTRA_OBJECTS)


def get_detection_categories(dataset='hssd', goal_category=None, split=None):
    """
    Get categories for GLIP/FastSAM detection prompts.
    """
    base_categories = get_categories(dataset, goal_category=goal_category, split=split)
    return base_categories + get_extra_objects(dataset)


def get_small_objects(dataset='hssd'):
    """
    Get HSSD small object categories in unified internal naming.
    """
    dataset = dataset.lower()
    if dataset == 'coin':
        return list(COIN_SMALL_OBJECTS)
    if dataset in OPEN_VOCAB_EPISODE_DATASETS:
        return []
    if dataset in ('hssd', 'instancenav'):
        return [HSSD_TO_UNIFIED.get(cat, cat) for cat in HSSD_SMALL_OBJECTS]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_threshold(category, dataset='hssd'):
    """
    Get HSSD detection threshold for a category.
    """
    dataset = dataset.lower()

    if dataset == 'coin':
        category = dataset_to_unified_name(category, dataset)
        return COIN_DETECTION_OVERRIDES.get(category, COIN_MIN_DETECTIONS)

    if dataset in OPEN_VOCAB_EPISODE_DATASETS:
        return 3
    if dataset in ('hssd', 'instancenav'):
        hssd_cat = UNIFIED_TO_HSSD.get(category, category)
        return HSSD_THRESHOLDS.get(hssd_cat, 3)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def unified_to_dataset_name(category, dataset='hssd', *, original_category=None):
    """
    Convert unified category name to HSSD category name.
    """
    dataset = dataset.lower()
    if dataset == 'coin':
        # Never guess an inverse for e.g. chair/armchair or clothes/hanging clothes.
        if original_category is not None:
            if dataset_to_unified_name(original_category, dataset) != dataset_to_unified_name(category, dataset):
                raise ValueError(f'CoIN category mismatch: {original_category!r} vs {category!r}')
            return original_category
        return category
    if dataset in OPEN_VOCAB_EPISODE_DATASETS:
        return category
    if dataset in ('hssd', 'instancenav'):
        return UNIFIED_TO_HSSD.get(category, category)
    raise ValueError(f"Unknown dataset: {dataset}")


def dataset_to_unified_name(category, dataset='hssd'):
    """
    Convert HSSD category name to unified category name.
    """
    dataset = dataset.lower()
    if dataset == 'coin':
        return COIN_TO_UNIFIED.get(category, category)
    if dataset in OPEN_VOCAB_EPISODE_DATASETS:
        return category
    if dataset in ('hssd', 'instancenav'):
        return HSSD_TO_UNIFIED.get(category, category)
    raise ValueError(f"Unknown dataset: {dataset}")


def get_category_count(dataset='hssd', goal_category=None, split=None):
    """Get the number of HSSD ObjectNav categories."""
    return len(get_categories(dataset, goal_category=goal_category, split=split))


def uses_episode_goal_categories(dataset='hssd'):
    """Whether this dataset should derive detection categories from each episode goal."""
    return is_open_vocabulary_dataset(dataset)
