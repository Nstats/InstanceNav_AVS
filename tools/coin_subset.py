#!/usr/bin/env python3
"""Write a seeded CoIN-Bench subset: a manifest plus a runtime yaml that uses it.

Episodes are drawn from distinct targets first (each CoIN target has about ten
start positions with one instruction), so a small subset covers as many target
instances as possible. Output goes to data/runtime/coin/ (gitignored). The yaml
stem is unique per (split, pool, count, seed) because PSG_Nav writes its runtime
copy to /tmp/psgnav_runtime_configs/<stem>.<split>.yaml.

--pool draws the subset only from the episodes of a frozen pool file, such as
configs/coin_val_unseen_subset100_seed77.json (tracked): count = pool size
gives exactly the pool. Pool-based stems carry the pool size
(coin_contextnav_val_unseen_pool100_n10_seed77), so they never overwrite a
subset of the whole split. --write-pool writes such a pool file from the whole
split with select().

  python tools/coin_subset.py --split val_seen --count 3 --seed 77
  python tools/coin_subset.py --split val_unseen --count 10 --pool configs/coin_val_unseen_subset100_seed77.json
  python tools/coin_subset.py --split val_unseen --count 100 --seed 77 \\
      --write-pool configs/coin_val_unseen_subset100_seed77.json
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'habitat-lab')]
DEFAULT_CONFIG = 'configs/coin_contextnav.yaml'
OUT_DIR = Path('data/runtime/coin')
POOL_FIELDS = ('split', 'selected_count', 'episode_keys')


def load_episodes(config_path, split):
    import habitat
    import utils.coin_dataset  # noqa: F401  (registers CoINInstanceNav-v1)
    config = habitat.get_config(config_path)
    config.defrost()
    config.DATASET.SPLIT = split
    config.DATASET.SUBSET_MANIFEST = ''
    config.freeze()
    return habitat.make_dataset(config.DATASET.TYPE, config=config.DATASET).episodes


def select(episodes, count, seed):
    """Seeded pick: one episode from each of `count` distinct targets while targets
    last, then more episodes from the remaining pool. Returns (scene, id) keys."""
    from utils.coin_dataset import episode_key
    if not 0 < count <= len(episodes):
        raise ValueError(f'count must be in 1..{len(episodes)}, got {count}')
    rng = random.Random(seed)
    by_target = defaultdict(list)
    for episode in episodes:
        by_target[(episode.scene_id, episode.object_instance_id)].append(episode)
    targets = sorted(by_target)
    rng.shuffle(targets)
    chosen = [rng.choice(by_target[target]) for target in targets[:count]]
    if count > len(targets):
        taken = {episode_key(ep) for ep in chosen}
        rest = [ep for ep in episodes if episode_key(ep) not in taken]
        chosen += rng.sample(rest, count - len(chosen))
    return sorted([list(episode_key(ep)) for ep in chosen])


def load_pool(path):
    """Read a frozen pool file; returns (manifest dict, set of (scene, id) keys)."""
    pool = json.loads(Path(path).read_text())
    missing = [name for name in POOL_FIELDS if name not in pool]
    if missing:
        raise ValueError(f'CoIN pool {path}: missing fields {missing}')
    keys = [tuple(key) for key in pool['episode_keys']]
    if len(set(keys)) != len(keys) or len(keys) != pool['selected_count']:
        raise ValueError(f'CoIN pool {path}: {len(keys)} keys ({len(set(keys))} unique), '
                         f'selected_count {pool["selected_count"]}')
    return pool, set(keys)


def restrict_to_pool(episodes, keys):
    """Episodes whose (scene, id) key is in `keys`, in split order. Every key must be found."""
    from utils.coin_dataset import episode_key
    keys = {tuple(key) for key in keys}
    kept = [ep for ep in episodes if episode_key(ep) in keys]
    found = {episode_key(ep) for ep in kept}
    if found != keys:
        absent = sorted(keys - found)
        raise ValueError(f'{len(absent)} pool keys are not in the split, first: {list(absent[0])}')
    return kept


def select_from_pool(episodes, keys, count, seed):
    """select() over the pool's episodes only; count == pool size returns the whole pool."""
    pool = restrict_to_pool(episodes, keys)
    if not 0 < count <= len(pool):
        raise ValueError(f'count must be in 1..{len(pool)} (pool size), got {count}')
    return select(pool, count, seed)


def pool_manifest(episodes, split, count, seed, config_path=DEFAULT_CONFIG):
    """Frozen pool: select() on the whole split, documented. Loads as a subset manifest."""
    keys = select(episodes, count, seed)
    chosen = {tuple(key) for key in keys}
    targets = {(ep.scene_id, ep.object_instance_id) for ep in restrict_to_pool(episodes, chosen)}
    return {'split': split, 'seed': seed, 'selected_count': len(keys), 'source_count': len(episodes),
            'source_config': str(config_path), 'targets': len(targets),
            'method': (f'tools/coin_subset.py select() with seed {seed} on the whole {split} split: '
                       'one episode per target, all targets, then random episodes from the rest.'),
            'episode_keys': keys}


def _dump_manifest(manifest):
    """Indented JSON with one episode key per line (diff-friendly for tracked files)."""
    head = {name: value for name, value in manifest.items() if name != 'episode_keys'}
    lines = json.dumps(head, indent=2)[:-2].splitlines()
    keys = [f'    {json.dumps(list(key))}' for key in manifest['episode_keys']]
    return '\n'.join(lines) + ',\n  "episode_keys": [\n' + ',\n'.join(keys) + '\n  ]\n}\n'


def write_pool(split, count, seed, path, config_path=DEFAULT_CONFIG):
    episodes = load_episodes(config_path, split)
    manifest = pool_manifest(episodes, split, count, seed, config_path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_manifest(manifest))
    return path, manifest


def write_subset(split, count, seed, config_path=DEFAULT_CONFIG, out_dir=OUT_DIR, pool_path=None):
    pool_keys = None
    if pool_path:
        pool, pool_keys = load_pool(pool_path)
        if pool['split'] != split:
            raise ValueError(f'CoIN pool {pool_path} is for split {pool["split"]}, not {split}')
    episodes = load_episodes(config_path, split)
    if pool_keys is None:
        keys, source_count, pool_tag = select(episodes, count, seed), len(episodes), ''
    else:
        keys = select_from_pool(episodes, pool_keys, count, seed)
        source_count, pool_tag = len(pool_keys), f'_pool{len(pool_keys)}'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{Path(config_path).stem}_{split}{pool_tag}_n{count}_seed{seed}'
    manifest_path = out_dir / f'{stem}.manifest.json'
    manifest = {'split': split, 'seed': seed, 'selected_count': len(keys), 'source_count': source_count,
                'source_config': str(config_path)}
    if pool_path:
        manifest['pool'] = str(pool_path)
    manifest['episode_keys'] = keys
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    lines, in_dataset = [], False
    for line in Path(config_path).read_text().splitlines():
        stripped = line.strip()
        if stripped == 'DATASET:':
            in_dataset = True
        elif in_dataset and stripped and not line.startswith(' '):
            in_dataset = False
        if in_dataset and stripped.startswith(('SPLIT:', 'SUBSET_MANIFEST:')):
            continue
        lines.append(line)
        if stripped == 'DATASET:':
            lines += [f'  SPLIT: {split}', f'  SUBSET_MANIFEST: {manifest_path}']
    yaml_path = out_dir / f'{stem}.yaml'
    yaml_path.write_text('\n'.join(lines) + '\n')
    return manifest_path, yaml_path, keys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', required=True, choices=('val_seen', 'val_seen_synonyms', 'val_unseen'))
    parser.add_argument('--count', required=True, type=int)
    parser.add_argument('--seed', default=77, type=int)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--out-dir', default=str(OUT_DIR))
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--pool', default=None, help='Frozen pool file; the subset is drawn from its episodes only')
    source.add_argument('--write-pool', default=None, metavar='PATH',
                        help='Write a frozen pool file (select() on the whole split) instead of a subset')
    args = parser.parse_args()
    try:
        if args.write_pool:
            path, manifest = write_pool(args.split, args.count, args.seed, args.write_pool, args.config)
            print(f'Pool: {path} ({manifest["selected_count"]}/{manifest["source_count"]} episodes, '
                  f'{manifest["targets"]} targets)')
            return 0
        manifest_path, yaml_path, keys = write_subset(args.split, args.count, args.seed, args.config,
                                                      args.out_dir, args.pool)
    except (OSError, ValueError) as exc:
        print(f'coin_subset: {exc}', file=sys.stderr)
        return 2
    print(f'Manifest: {manifest_path} ({len(keys)} episodes)')
    print(f'Config: {yaml_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
