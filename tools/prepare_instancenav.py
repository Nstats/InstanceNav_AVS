#!/usr/bin/env python3
"""Prepare PSL InstanceNav val episodes with instance-specific goals and attributes."""
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import random


def read_gzip(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def prepare(attributes_path, content_dir, output, fraction=0.2, seed=77):
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0, 1]')
    attributes_path, content_dir, output = map(Path, (attributes_path, content_dir, output))
    attributes = read_gzip(attributes_path)['attribute_data']
    episodes, goals, missing = [], {}, []
    sources = sorted(content_dir.glob('*.json.gz'))
    if not sources:
        raise FileNotFoundError(f'No episode shards in {content_dir}')
    categories = set()
    fingerprints = {str(attributes_path): hashlib.sha256(attributes_path.read_bytes()).hexdigest()}
    for source in sources:
        fingerprints[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        data = read_gzip(source)
        for raw in data['episodes']:
            episode = copy.deepcopy(raw)
            scene = Path(episode['scene_id']).name.split('.')[0]
            key = f"{scene}_{episode['goal_object_id']}"
            if key not in attributes:
                missing.append(key)
                continue
            if key not in data['goals']:
                raise ValueError(f'Missing instance goal: {key}')
            goal = data['goals'][key]
            if not goal.get('view_points'):
                raise ValueError(f'Missing target viewpoints: {key}')
            episode['goal_attributes'] = attributes[key]
            # Preserve the source ID; never replace it with a per-shard counter.
            episode['episode_id'] = str(episode['episode_id'])
            episode['goals'] = []
            goals[key] = goal
            categories.add(episode['object_category'])
            episodes.append(episode)
    if missing:
        raise ValueError(f'{len(missing)} episodes lack PSL descriptions; refusing a biased subset')
    episodes.sort(key=lambda ep: (ep['scene_id'], ep['episode_id'], str(ep['goal_object_id'])))
    identities = [(ep['scene_id'], ep['episode_id']) for ep in episodes]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate episode identities')
    random.Random(seed).shuffle(episodes)
    count = math.floor(len(episodes) * fraction)
    if count < 1:
        raise ValueError('Subset is empty')
    selected = episodes[:count]
    keys = {Path(ep['scene_id']).name.split('.')[0] + '_' + str(ep['goal_object_id']) for ep in selected}
    manifest = {
        'dataset': 'PSL-InstanceNav', 'split': 'val',
        'split_note': 'Official HM3D validation scenes (unseen); no split named val_unseen.',
        'fraction': fraction, 'seed': seed, 'sampling_unit': 'episode',
        'sampling': 'sort(scene_id, episode_id, goal_object_id), Random(seed).shuffle, floor(N*fraction)',
        'source_count': len(episodes), 'selected_count': count,
        'source_sha256': fingerprints,
        'scenes': sorted({ep['scene_id'] for ep in selected}),
        'episodes': [{k:ep[k] for k in ('scene_id','episode_id','goal_object_id','object_category')} for ep in selected],
    }
    payload = {
        'episodes': selected, 'goals': {key:goals[key] for key in sorted(keys)},
        'category_to_task_category_id': {cat:i for i,cat in enumerate(sorted(categories))},
        'subset': {k:v for k,v in manifest.items() if k not in ('episodes','source_sha256','scenes')},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
            gz.write(json.dumps(payload, ensure_ascii=False).encode())
    manifest['prepared_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('').with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attributes', default='data/datasets/instancenav/val/val_text.json.gz')
    parser.add_argument('--content', default='data/datasets/instance_imagenav_hm3d_v3/val/content')
    parser.add_argument('--output', default='data/datasets/instancenav/val20/val20.json.gz')
    parser.add_argument('--fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=77)
    args = parser.parse_args()
    result = prepare(args.attributes, args.content, args.output, args.fraction, args.seed)
    print(f"Selected {result['selected_count']}/{result['source_count']} episodes in {len(result['scenes'])} scenes; seed={args.seed}")
