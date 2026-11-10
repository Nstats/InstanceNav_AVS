#!/usr/bin/env python3
"""Download public InstanceNav val data and optionally authorized HM3D val scenes.

Credentials are loaded from PSGNAV_ENV_FILE/.env, never from command-line arguments.
"""
import argparse
import gzip
import os
from pathlib import Path
import shutil
import sys
import tarfile
import zipfile

import requests
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.model_client import load_environment


def download(url, target, auth=None):
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        return target
    part=target.with_suffix(target.suffix+'.part')
    with requests.get(url,auth=auth,stream=True,timeout=(15,120)) as response:
        if not response.ok:
            raise RuntimeError(f'Download HTTP {response.status_code}: {target.name}')
        if 'text/html' in response.headers.get('Content-Type','').lower():
            raise RuntimeError(f'{target.name}: server returned a webpage; check Matterport authorization/download availability')
        with part.open('wb') as output:
            for chunk in response.iter_content(1024*1024): output.write(chunk)
    part.replace(target)
    return target


def extract_tar(archive, destination):
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            relative=Path(member.name)
            if relative.is_absolute() or '..' in relative.parts or member.issym() or member.islnk():
                raise ValueError(f'Unsafe tar member in {archive.name}')
        tar.extractall(destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example',action='store_true',help='Download the official public validation example for smoke testing')
    parser.add_argument('--hm3d',action='store_true',help='Also download authorized HM3D v0.2 val assets')
    args=parser.parse_args()
    os.chdir(ROOT); load_environment()
    downloads=Path('data/downloads')
    attrs=Path('data/datasets/instancenav/val/val_text.json.gz')
    download('https://drive.google.com/uc?export=download&id=1KNdv6isX1FDZi4KCVPiECYDxijg9cZ3L',attrs)
    with gzip.open(attrs,'rt') as f: f.read()
    archive=download('https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip',downloads/'instance_imagenav_hm3d_v3.zip')
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.startswith('instance_imagenav_hm3d_v3/val/'):
                if '..' in Path(name).parts: raise ValueError('Unsafe zip path')
                z.extract(name,'data/datasets')
    from tools.prepare_instancenav import prepare
    manifest=prepare(attrs,'data/datasets/instance_imagenav_hm3d_v3/val/content','data/datasets/instancenav/val20/val20.json.gz')
    print(f"Prepared unseen validation subset: {manifest['selected_count']}/{manifest['source_count']} episodes")
    if args.example:
        import json
        import hashlib
        root=Path('data/scene_datasets/hm3d_v0.2')
        for filename in ('hm3d-example-habitat-v0.2.tar','hm3d-example-semantic-annots-v0.2.tar'):
            archive=download('https://media.githubusercontent.com/media/matterport/habitat-matterport-3dresearch/main/example/'+filename,downloads/filename)
            if filename == 'hm3d-example-habitat-v0.2.tar':
                if hashlib.sha256(archive.read_bytes()).hexdigest() != 'bbafcfb13e84f404e95fb99022f0d6702571be795f3f711ad396d0bae5b33ef7':
                    raise RuntimeError('Public example checksum mismatch')
            with tarfile.open(archive) as tar:
                members=[m for m in tar.getmembers() if m.name.startswith('00861-GLAQ4DNUx5U/')]
                if any('..' in Path(m.name).parts or m.issym() or m.islnk() for m in members):
                    raise ValueError('Unsafe example archive')
                (root/'val').mkdir(parents=True,exist_ok=True)
                tar.extractall(root/'val',members=members)
        archive=download('https://raw.githubusercontent.com/matterport/habitat-matterport-3dresearch/main/example/hm3d-example-semantic-configs-v0.2.tar',downloads/'hm3d-example-semantic-configs-v0.2.tar')
        with tarfile.open(archive) as tar:
            config_path=root/'hm3d_annotated_basis.scene_dataset_config.json'
            if not config_path.exists():
                config_path.write_bytes(tar.extractfile('hm3d_annotated_basis.scene_dataset_config.json').read())
        (root/'example').mkdir(exist_ok=True)
        link=root/'example/00861-GLAQ4DNUx5U'
        if not link.exists():link.symlink_to('../val/00861-GLAQ4DNUx5U',target_is_directory=True)
        with gzip.open('data/datasets/instancenav/val20/val20.json.gz','rt') as f:
            smoke=json.load(f)
        smoke['episodes']=[next(ep for ep in smoke['episodes'] if 'GLAQ4DNUx5U' in ep['scene_id'])]
        smoke_dir=Path('data/datasets/instancenav/smoke');smoke_dir.mkdir(exist_ok=True)
        with gzip.open(smoke_dir/'smoke.json.gz','wt') as f:json.dump(smoke,f)
        print('Public unseen example ready for tools/smoke_navigation.py; not the full 20% benchmark')
    if args.hm3d:
        key=os.environ.get('MATTERPORT_TOKEN_ID');secret=os.environ.get('MATTERPORT_TOKEN_SECRET')
        if not key or not secret:
            raise RuntimeError('HM3D requires MATTERPORT_TOKEN_ID and MATTERPORT_TOKEN_SECRET in the environment or PSGNAV_ENV_FILE')
        root=Path('data/scene_datasets/hm3d_v0.2')
        for filename in ('hm3d-val-habitat-v0.2.tar','hm3d-val-configs.tar',
                         'hm3d-val-semantic-annots-v0.2.tar','hm3d-val-semantic-configs-v0.2.tar'):
            archive=download('https://api.matterport.com/resources/habitat/'+filename,downloads/filename,auth=(key,secret))
            extract_tar(archive,root/'val')
        for path in (root/'val').glob('*.scene_dataset_config.json'):
            shutil.copyfile(path,root/path.name)
        print('HM3D val download complete; run tools/preflight_instancenav.py')


if __name__=='__main__':
    try: main()
    except (RuntimeError,requests.RequestException) as exc:
        # Do not echo authenticated URLs or request headers.
        print(str(exc) if isinstance(exc,RuntimeError) else 'Download network failure',file=sys.stderr)
        raise SystemExit(1)
