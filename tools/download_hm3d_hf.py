#!/usr/bin/env python3
"""Download only the fixed InstanceNav subset's HM3D assets from a pinned HF mirror."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = 'Tsaisplus/mp3d_hm3d_cleanval-scene'
REVISION = '0c6e40bfe6bac20de79b74fe2e61e7823e4f325b'


def valid(path, item):
    if not path.is_file() or path.stat().st_size != item['size']:
        return False
    if 'lfs' in item:
        digest = hashlib.sha256()
        expected = item['lfs']['sha256']
    else:
        digest = hashlib.sha1(f"blob {item['size']}\0".encode())
        expected = item['blobId']
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proxy', default='http://127.0.0.1:7897')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    proxies = {'http': args.proxy, 'https': args.proxy} if args.proxy else {}
    response = requests.get(f'https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}',
                            params={'blobs': 'true'}, proxies=proxies, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    manifest = json.loads((ROOT / 'data/datasets/instancenav/val20/val20.manifest.json').read_text())
    scene_dirs = {Path(scene).parent.name for scene in manifest['scenes']}
    selected = [item for item in metadata['siblings']
                if any(f'/{scene}/' in item['rfilename'] for scene in scene_dirs)
                and item['rfilename'].endswith(('.basis.glb', '.basis.navmesh', '.semantic.glb', '.semantic.txt'))]
    for scene in scene_dirs:
        if len([item for item in selected if f'/{scene}/' in item['rfilename']]) != 4:
            raise RuntimeError(f'Incomplete scene in mirror: {scene}')
    selected += [item for item in metadata['siblings'] if item['rfilename'] in (
        'scene_hm3d/hm3d_annotated_basis.scene_dataset_config.json',
        'scene_hm3d/val/hm3d_annotated_val_basis.scene_dataset_config.json',
        'scene_hm3d/val/hm3d_val_basis.scene_dataset_config.json')]
    target = ROOT / 'data/scene_datasets/hm3d_v0.2'
    records = ROOT / 'data/downloads/hm3d_hf_manifest.json'
    records.parent.mkdir(parents=True, exist_ok=True)
    records.write_text(json.dumps({'repo': REPO, 'revision': REVISION, 'files': selected}, indent=2))
    print(f'{len(scene_dirs)} scenes, {len(selected)} files, {sum(i["size"] for i in selected)/1e9:.3f} GB', flush=True)

    def download(item):
        name = item['rfilename']
        relative = Path(name).relative_to('scene_hm3d')
        path = target / relative
        if valid(path, item):
            return f'VERIFIED {relative}'
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + '.part')
        for attempt in range(5):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == item['size'] and valid(partial, item):
                    partial.replace(path)
                    return f'DOWNLOADED {relative}'
                if offset >= item['size']:
                    partial.unlink()
                    offset = 0
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}'
                with requests.get(url, headers=headers, proxies=proxies, stream=True, timeout=(30, 90)) as res:
                    res.raise_for_status()
                    append = offset and res.status_code == 206
                    if append and not res.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise RuntimeError('Invalid resume range')
                    with partial.open('ab' if append else 'wb') as stream:
                        for chunk in res.iter_content(1024 * 1024):
                            stream.write(chunk)
                if not valid(partial, item):
                    raise RuntimeError(f'Size/hash mismatch: {relative}')
                partial.replace(path)
                return f'DOWNLOADED {relative}'
            except (requests.RequestException, RuntimeError) as exc:
                # Do not log redirect URLs; these can contain signed credentials.
                print(f'RETRY {relative}: {type(exc).__name__} attempt={attempt+1}', flush=True)
                if attempt == 4:
                    raise RuntimeError(f'Download failed: {relative}') from None
                time.sleep(2 ** attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, item) for item in selected]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            print(f'[{index}/{len(selected)}] {future.result()}', flush=True)
    print('HM3D_HF_DOWNLOAD_OK: all files verified against pinned HF metadata', flush=True)


if __name__ == '__main__':
    main()
