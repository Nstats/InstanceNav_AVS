#!/usr/bin/env python3
"""Manual, one-experiment launcher for the five planned InstanceNav ablations.

--dry-run prints only; --check-only imports dependencies but never navigates or
calls a model. Normal execution archives the configuration and owned runtime
source, then delegates to run_instance100.sh. There is deliberately no 'all'.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
COMMON = {
    'LANDMARK_COMPARISON_REPEATS': '1', 'LANDMARK_COMPARISON_WORKERS': '10',
    'ENABLE_INSTANCE_DESCRIPTION_FOR_EXPLORATION': 'false',
    'ENABLE_OBJECT_RELATIONS': 'false', 'ENABLE_GROUP_LLM_FILTER': 'true',
    'ENABLE_ROOM_LLM_FILTER': 'true', 'ENABLE_INSTANCE_CONFIRMATION': 'true',
    'INSTANCE_CONFIRMATION_THRESHOLD': '1', 'VERIFY_BEFORE_CANDIDATE': 'false',
    'INSTANCE_CONFIRMATION_COUNT_UNIT': 'frame',
    'INSTANCE_CONFIRMATION_MODE': 'positive_match',
    'INSTANCE_VERIFICATION_PROMPT': 'match_strict_no',
    'INSTANCE_VERIFICATION_SCOPE': 'instance', 'INSTANCE_REJECTION_POLICY': 'none',
    'INSTANCE_VERIFICATION_RECORD_INPUTS': 'true',
    'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK': 'false',
    'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY': 'false', 'INSTANCE_COLLECT_ONLY': 'false',
    'ENABLE_RELATIVE_EVIDENCE': 'false', 'RAG_MODE': '0', 'VLM': 'qwen3.5-flash',
}
VARIANTS = {
    'B': {},
    'E1': {'ENABLE_INSTANCE_CONFIRMATION': 'false'},
    'E2': {'INSTANCE_VERIFICATION_SCOPE': 'category'},
    'E3': {'INSTANCE_REJECTION_POLICY': 'no'},
    'E4': {'INSTANCE_REJECTION_POLICY': 'non_yes'},
    'E5': {'INSTANCE_CONFIRMATION_THRESHOLD': '2'},
}


def settings(experiment):
    return {**COMMON, **VARIANTS[experiment]}


def launch_command(experiment, gpus, timestamp=None, resume=False):
    cmd = ['bash', str(ROOT / 'scripts/run_instance100.sh')]
    for key, value in settings(experiment).items():
        cmd.extend(['--' + key.lower().replace('_', '-'), value])
    cmd.extend(['--gpus', gpus])
    if timestamp:
        cmd.extend(['--timestamp', timestamp])
    if resume:
        cmd.append('--resume')
    return cmd


def source_inventory():
    """Archive application code/config, not credentials, datasets or model weights."""
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT
    ).decode().split('\0')
    paths = set()
    for name in filter(None, names):
        p = Path(name)
        if ((len(p.parts) == 1 and p.suffix in {'.py', '.yml', '.yaml'}) or
                (p.parts[0] in {'utils', 'configs', 'graph', 'scripts', 'tools'}
                 and p.suffix in {'.py', '.sh', '.json', '.yaml', '.yml'})):
            if (ROOT / p).is_file():
                paths.add(name)
    # The subset itself is hashed, not archived (large data remains external).
    data = ['data/datasets/instancenav/val10/val10.json.gz',
            'data/datasets/instancenav/val10/val10.manifest.json']
    hashes = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
              for n in sorted(paths | set(data))}
    submodules = subprocess.check_output(['git', 'submodule', 'status', '--recursive'], cwd=ROOT).decode()
    digest = hashlib.sha256(json.dumps({'files': hashes, 'submodules': submodules},
                                    sort_keys=True).encode()).hexdigest()
    return sorted(paths), hashes, submodules, digest


def archive_run(args, cmd, directory, inventory):
    paths, hashes, submodules, digest = inventory
    manifest_path = directory / 'manifest.json'
    identity = {'experiment': args.experiment, 'settings': settings(args.experiment),
                'source_digest': digest, 'python': str(Path(sys.executable).resolve()),
                'gpus': args.gpus, 'timestamp': args.timestamp}
    if args.resume:
        if not manifest_path.is_file():
            raise ValueError('Resume requires this launcher\'s original manifest')
        prior = json.loads(manifest_path.read_text())
        if any(prior.get(k) != v for k, v in identity.items()):
            raise ValueError('Resume source/configuration/interpreter/GPUs changed; restore the original run settings')
        return manifest_path
    if directory.exists() or (ROOT / f'data/results_instancenav_{args.timestamp}').exists():
        raise ValueError('Timestamp already exists; use --resume or a new timestamp')
    suite_path = ROOT / 'outputs/instance_ablation_runs' / args.suite / 'suite.json'
    suite_identity = {'source_digest': digest, 'python': identity['python'], 'common': COMMON}
    if suite_path.exists():
        if json.loads(suite_path.read_text()) != suite_identity:
            raise ValueError('Suite source/configuration changed; restore it or select a new --suite for a new comparison batch')
    directory.mkdir(parents=True)
    if not suite_path.exists():
        suite_path.write_text(json.dumps(suite_identity, indent=2) + '\n')
    with tarfile.open(directory / 'source.tar.gz', 'w:gz') as archive:
        for name in paths:
            archive.add(ROOT / name, arcname=name, recursive=False)
    manifest = {**identity, 'command': cmd, 'files_sha256': hashes,
                'submodule_versions': submodules, 'python_version': sys.version,
                'archive_scope': 'Owned application source/config; dependency and dataset paths remain external.',
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=VARIANTS)
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--timestamp')
    parser.add_argument('--suite', default='avs5_20261110')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for value in (args.suite, args.timestamp):
        if value is not None and not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            parser.error('suite/timestamp must contain only letters, digits, underscore or hyphen')
    if not re.fullmatch(r'\d+,\d+', args.gpus) or len(set(args.gpus.split(','))) != 2:
        parser.error('--gpus requires two distinct IDs, e.g. 0,1')
    if not args.timestamp and (args.resume or not (args.check_only or args.dry_run)):
        parser.error('--timestamp is required for evaluation/resume')
    cmd = launch_command(args.experiment, args.gpus, args.timestamp, args.resume)
    if args.dry_run:
        print(json.dumps({'experiment': args.experiment, 'settings': settings(args.experiment),
                          'command': cmd}, indent=2))
        return 0
    env = {**os.environ, **settings(args.experiment), 'PSGNAV_PYTHON': sys.executable,
           'INSTANCE_ABLATION_ID': args.experiment}
    for legacy in ('ENABLE_CONTRADICTION_CHECK', 'ENABLE_INSTANCE_CONFIRMATION_BEFORE_FIND_GOAL'):
        env.pop(legacy, None)
    print('Experiment:', args.experiment, '\nCommand:', shlex.join(cmd), flush=True)
    result = subprocess.run([*cmd, '--check-only'], env=env, cwd=ROOT)
    if result.returncode or args.check_only:
        return result.returncode
    directory = ROOT / 'outputs/instance_ablation_runs' / args.suite / args.timestamp
    manifest = archive_run(args, cmd, directory, source_inventory())
    env['INSTANCE_ABLATION_MANIFEST'] = str(manifest)
    print('Manifest:', manifest, flush=True)
    os.chdir(ROOT)
    os.execvpe('bash', cmd, env)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ValueError as exc:
        raise SystemExit(str(exc))
