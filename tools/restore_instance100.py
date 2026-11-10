#!/usr/bin/env python3
"""Rebuild the frozen InstanceNav 100-task subset and verify its exact identity."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

try:
    from .prepare_instancenav import prepare, read_gzip
except ImportError:
    from prepare_instancenav import prepare, read_gzip

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs/runtime/val10.manifest.json'
ORDER = ROOT / 'configs/runtime/evaluation_order_fixed100.json'


def verify(output, manifest_path=MANIFEST, order_path=ORDER):
    output = Path(output)
    manifest = json.loads(Path(manifest_path).read_text())
    if hashlib.sha256(output.read_bytes()).hexdigest() != manifest['prepared_sha256']:
        raise ValueError('Prepared dataset SHA256 differs from the frozen manifest')
    episodes = read_gzip(output)['episodes']
    identities = [{k: ep[k] for k in ('scene_id', 'episode_id', 'goal_object_id',
                                     'object_category')} for ep in episodes]
    if len(episodes) != manifest['selected_count'] or identities != manifest['episodes']:
        raise ValueError('Episode identities/order differ from the frozen manifest')
    # Match Habitat EpisodeIterator(group_by_scene=True, shuffle=False).
    scene_keys = {}
    for ep in episodes:
        scene_keys.setdefault(ep['scene_id'], len(scene_keys))
    grouped = sorted(episodes, key=lambda ep: scene_keys[ep['scene_id']])
    def identity(ep):
        return (ep['scene_id'].removeprefix('data/scene_datasets/'),
                str(ep['episode_id']), str(ep['goal_object_id']))
    order = json.loads(Path(order_path).read_text())
    if ([identity(ep) for ep in grouped] != [identity(ep) for ep in order]
            or [ep['index'] for ep in order] != list(range(len(episodes)))):
        raise ValueError('Grouped evaluation order differs from the frozen index')
    return manifest


def restore(attributes, content, output, manifest_path=MANIFEST, order_path=ORDER):
    output = Path(output)
    manifest = json.loads(Path(manifest_path).read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate before replacing an existing local dataset. No navigation or API calls.
    with tempfile.TemporaryDirectory(prefix='.restore-instance100-', dir=output.parent) as work:
        candidate = Path(work) / 'val10.json.gz'
        generated = prepare(attributes, content, candidate,
                            fraction=manifest['fraction'], seed=manifest['seed'])
        if generated['source_count'] != manifest['source_count']:
            raise ValueError('Source episode population differs from the frozen manifest')
        verify(candidate, manifest_path, order_path)
        candidate.replace(output)
    # Keep the canonical provenance identical across machines, including its source hashes.
    output.with_suffix('').with_suffix('.manifest.json').write_bytes(Path(manifest_path).read_bytes())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attributes', default=ROOT / 'data/datasets/instancenav/val/val_text.json.gz')
    parser.add_argument('--content', default=ROOT / 'data/datasets/instance_imagenav_hm3d_v3/val/content')
    parser.add_argument('--output', default=ROOT / 'data/datasets/instancenav/val10/val10.json.gz')
    parser.add_argument('--verify-only', action='store_true', help='Verify an existing subset; do not write files')
    args = parser.parse_args()
    try:
        result = (verify(args.output) if args.verify_only else
                  restore(args.attributes, args.content, args.output))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Fixed-100 verification failed: {exc}\n')
    print(f"Verified {result['selected_count']} tasks, {len(result['scenes'])} scenes; "
          f"dataset SHA256 and fixed evaluation order match Git.\n{args.output}")


if __name__ == '__main__':
    main()
