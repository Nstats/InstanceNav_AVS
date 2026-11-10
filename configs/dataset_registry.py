"""
Central registry for supported evaluation datasets.
"""

DATASET_CONFIGS = {
    "instancenav": {
        "config_path": "configs/instancenav_val20_qwen.yaml",
        "default_split": "val",
        "fallback_scene_count": None,
        "open_vocabulary": False,
    },
    "hm3d_ovon": {
        "config_path": "configs/hm3d_ovon_qwen.yaml",
        "default_split": "val_unseen",
        "fallback_scene_count": None,
        "open_vocabulary": True,
    },
    "hssd": {
        "config_path": "configs/challenge_objectnav_hssd.local.rgbd.yaml",
        "default_split": "val",
        "fallback_scene_count": 40,
        "open_vocabulary": False,
    },
    # CoIN-Bench under Context-Nav's protocol (full instruction, no user):
    # configs/coin_contextnav.yaml and data/datasets/CoIN/README.md.
    "coin": {
        "config_path": "configs/coin_contextnav.yaml",
        "default_split": "val_unseen",
        "fallback_scene_count": None,
        "open_vocabulary": True,
        # Goal detections closer than this (m) count as the same candidate.
        "goal_merge_threshold_m": 0.25,
    },
}

# Merge radius (m) for datasets without a goal_merge_threshold_m entry. InstanceNav,
# HSSD and HM3D-OVON were tuned and validated with it.
DEFAULT_GOAL_MERGE_THRESHOLD_M = 0.8

SUPPORTED_DATASETS = tuple(DATASET_CONFIGS.keys())

# Datasets whose goal is one described object instance, so instance
# verification applies. Every other dataset keeps category-level ObjectNav.
INSTANCE_DATASETS = ('instancenav', 'coin')


def is_instance_dataset(dataset) -> bool:
    """True for instance-goal datasets; equals ``dataset == 'instancenav'`` for all others."""
    return dataset in INSTANCE_DATASETS


def get_dataset_info(dataset: str):
    dataset = dataset.lower()
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset}. Supported datasets: {list(SUPPORTED_DATASETS)}"
        )
    return DATASET_CONFIGS[dataset]


def get_dataset_config_path(dataset: str) -> str:
    return get_dataset_info(dataset)["config_path"]


def get_default_dataset_split(dataset: str) -> str:
    return get_dataset_info(dataset)["default_split"]


def get_fallback_scene_count(dataset: str):
    return get_dataset_info(dataset)["fallback_scene_count"]


def is_open_vocabulary_dataset(dataset: str) -> bool:
    return bool(get_dataset_info(dataset)["open_vocabulary"])


def get_goal_merge_threshold(dataset: str) -> float:
    """Distance (m) within which goal detections merge into one candidate (0.8 unless set)."""
    return get_dataset_info(dataset).get("goal_merge_threshold_m", DEFAULT_GOAL_MERGE_THRESHOLD_M)
