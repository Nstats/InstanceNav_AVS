"""CoIN subsets: the frozen val_unseen pool and subsets drawn from it (tools/coin_subset.py)."""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.coin_subset as cs

ROOT = Path(__file__).resolve().parents[1]
POOL = 'configs/coin_val_unseen_subset100_seed77.json'
CONFIG = str(ROOT / 'configs/coin_contextnav.yaml')
COIN_ROOT = ROOT / 'data/datasets/CoIN/CoIN-Bench'


def _episodes(targets=6, starts=4):
    return [SimpleNamespace(scene_id=f'data/scene_datasets/hm3d_v0.2/val/s{t % 3}/s{t % 3}.basis.glb',
                            object_instance_id=f'obj_{t}', episode_id=str(10 * t + k))
            for t in range(targets) for k in range(starts)]


def _key(episode):
    return Path(episode.scene_id).name, episode.episode_id


def _pool_keys(episodes):
    # Two starts of every target plus one extra target's four starts: 14 of 24 episodes.
    return {_key(ep) for ep in episodes if int(ep.episode_id) % 10 < 2 or ep.object_instance_id == 'obj_5'}


def test_restrict_to_pool_keeps_split_order_and_rejects_unknown_keys():
    episodes = _episodes()
    keys = _pool_keys(episodes)
    kept = cs.restrict_to_pool(episodes, keys)
    assert {_key(ep) for ep in kept} == keys
    assert kept == [ep for ep in episodes if _key(ep) in keys]
    assert cs.restrict_to_pool(episodes, [list(key) for key in keys]) == kept  # JSON lists work too
    with pytest.raises(ValueError, match='not in the split'):
        cs.restrict_to_pool(episodes, keys | {('s0.basis.glb', '999')})


def test_pool_size_returns_the_whole_pool_and_larger_counts_fail():
    episodes = _episodes()
    keys = _pool_keys(episodes)
    assert cs.select_from_pool(episodes, keys, len(keys), seed=77) == sorted(map(list, keys))
    with pytest.raises(ValueError, match='pool size'):
        cs.select_from_pool(episodes, keys, len(keys) + 1, seed=77)
    with pytest.raises(ValueError):
        cs.select_from_pool(episodes, keys, 0, seed=77)


def test_smaller_counts_are_deterministic_subsets_of_the_pool():
    episodes = _episodes()
    keys = _pool_keys(episodes)
    # Episodes outside the pool do not change the pick.
    assert cs.select_from_pool(episodes, keys, 5, seed=3) == cs.select(cs.restrict_to_pool(episodes, keys), 5, 3)
    for count in (1, 5, 6, 9):
        picked = cs.select_from_pool(episodes, keys, count, seed=3)
        assert picked == cs.select_from_pool(episodes, keys, count, seed=3)
        assert len(picked) == count and {tuple(key) for key in picked} <= keys
    by_key = {_key(ep): ep for ep in episodes}
    six = cs.select_from_pool(episodes, keys, 6, seed=3)
    assert len({by_key[tuple(key)].object_instance_id for key in six}) == 6  # distinct targets first


def test_write_subset_from_pool(tmp_path, monkeypatch):
    episodes = _episodes()
    keys = _pool_keys(episodes)
    pool_path = tmp_path / 'pool.json'
    pool_path.write_text(json.dumps({'split': 'val_unseen', 'seed': 77, 'selected_count': len(keys),
                                     'episode_keys': sorted(map(list, keys))}))
    monkeypatch.setattr(cs, 'load_episodes', lambda config_path, split: episodes)
    manifest_path, yaml_path, picked = cs.write_subset('val_unseen', 4, 77, CONFIG, tmp_path / 'out',
                                                       pool_path=str(pool_path))
    assert manifest_path.name == 'coin_contextnav_val_unseen_pool14_n4_seed77.manifest.json'
    assert yaml_path.name == 'coin_contextnav_val_unseen_pool14_n4_seed77.yaml'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['pool'] == str(pool_path) and manifest['source_count'] == len(keys)
    assert manifest['episode_keys'] == picked == cs.select_from_pool(episodes, keys, 4, 77)
    assert f'SUBSET_MANIFEST: {manifest_path}' in yaml_path.read_text()
    with pytest.raises(ValueError, match='not val_seen'):
        cs.write_subset('val_seen', 4, 77, CONFIG, tmp_path / 'out', pool_path=str(pool_path))
    with pytest.raises(ValueError, match='pool size'):
        cs.write_subset('val_unseen', len(keys) + 1, 77, CONFIG, tmp_path / 'out', pool_path=str(pool_path))
    whole, _, _ = cs.write_subset('val_unseen', 4, 77, CONFIG, tmp_path / 'out')
    assert whole.name == 'coin_contextnav_val_unseen_n4_seed77.manifest.json'
    assert 'pool' not in json.loads(whole.read_text())


def test_write_pool_round_trips(tmp_path, monkeypatch):
    episodes = _episodes()
    monkeypatch.setattr(cs, 'load_episodes', lambda config_path, split: episodes)
    path, manifest = cs.write_pool('val_unseen', 10, 77, tmp_path / 'pool.json')
    loaded = json.loads(path.read_text())
    assert loaded == manifest
    assert loaded['episode_keys'] == cs.select(episodes, 10, 77)
    assert (loaded['targets'], loaded['source_count'], loaded['selected_count']) == (6, 24, 10)
    _, keys = cs.load_pool(path)
    assert cs.select_from_pool(episodes, keys, 10, 77) == loaded['episode_keys']


def test_load_pool_rejects_inconsistent_files(tmp_path):
    path = tmp_path / 'pool.json'
    path.write_text(json.dumps({'split': 'val_unseen', 'selected_count': 2, 'episode_keys': [['a', '1'], ['a', '1']]}))
    with pytest.raises(ValueError, match='unique'):
        cs.load_pool(path)
    path.write_text(json.dumps({'split': 'val_unseen', 'episode_keys': [['a', '1']]}))
    with pytest.raises(ValueError, match='selected_count'):
        cs.load_pool(path)


def test_frozen_pool_file():
    pool, keys = cs.load_pool(ROOT / POOL)
    assert pool['split'] == 'val_unseen' and pool['seed'] == 77
    assert pool['selected_count'] == len(keys) == len(pool['episode_keys']) == 100
    assert pool['source_count'] == 459 and pool['targets'] == 47
    assert pool['episode_keys'] == sorted(pool['episode_keys'])


def test_launcher_uses_the_frozen_pool():
    script = (ROOT / 'scripts/run_coin.sh').read_text()
    assert re.search(r'^val_unseen_pool="([^"]+)"$', script, re.M).group(1) == POOL
    assert re.search(r'^: "\$\{COIN_SUBSET:=(\d+)\}"', script, re.M).group(1) == '100'


@pytest.mark.skipif(not (COIN_ROOT / 'val_unseen').is_dir(), reason='CoIN-Bench val_unseen not downloaded')
def test_frozen_pool_matches_the_real_split():
    import utils.coin_dataset as cd
    config = SimpleNamespace(DATA_PATH=str(COIN_ROOT / '{split}' / '{split}.json.gz'), SPLIT='val_unseen',
                             SCENES_DIR=str(ROOT / 'data/scene_datasets'))
    full = cd.CoINDataset(config).episodes
    _, keys = cs.load_pool(ROOT / POOL)
    pool = cs.restrict_to_pool(full, keys)  # every key is in the split
    assert len({(ep.scene_id, ep.object_instance_id) for ep in pool}) == 47
    assert len({(ep.scene_id, ep.object_instance_id) for ep in full}) == 47
    # Reproducible from the whole split, and loads unchanged as a subset manifest.
    assert cs.select(full, 100, 77) == sorted(map(list, keys))
    assert cs.select_from_pool(full, keys, 100, 77) == sorted(map(list, keys))
    subset = cd.CoINDataset(SimpleNamespace(**vars(config), SUBSET_MANIFEST=str(ROOT / POOL))).episodes
    assert {cd.episode_key(ep) for ep in subset} == keys
