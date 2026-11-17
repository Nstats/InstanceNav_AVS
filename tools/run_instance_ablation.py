#!/usr/bin/env python3
"""Manual, one-experiment launcher for the planned InstanceNav ablations.

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
# The queue script only lists which launcher commands to run next. Editing it to queue
# more runs must not change the fingerprint of the code under test, so it is neither
# hashed nor archived. Nothing it contains reaches the navigation process.
FINGERPRINT_EXCLUDE = {'scripts/run_continuous_evaluation.sh'}
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
    'INSTANCE_REJECTION_SCOPE': 'episode', 'ENABLE_CANDIDATE_APPROACH': 'true',
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
    # M7 (paper Table A, how far one negative answer reaches): a no/unknown releases the
    # candidate for the current approach only; it is not excluded (B: this frame only,
    # E3/E4: the whole episode).
    'T0': {'INSTANCE_REJECTION_POLICY': 'non_yes', 'INSTANCE_REJECTION_SCOPE': 'approach'},
    'T0no': {'INSTANCE_REJECTION_POLICY': 'no', 'INSTANCE_REJECTION_SCOPE': 'approach'},
    # M7 (paper Table C, which candidates are approached): none (A0) or only those the
    # VLM admits from where they are first seen (RA); B approaches every detection.
    'A0': {'ENABLE_CANDIDATE_APPROACH': 'false'},
    'RA': {'VERIFY_BEFORE_CANDIDATE': 'true'},
}
# Model behind every call (exploration and verification), as run_instance100.sh --vlm accepts it.
# The default keeps the existing presets unchanged; another VLM is a separate comparison batch.
VLMS = ('qwen3.5-flash', 'gemini-2.5-flash', 'qwen3-vl-8b-instruct')


def common(vlm=COMMON['VLM']):
    if vlm not in VLMS:
        raise ValueError(f'VLM must be one of {", ".join(VLMS)}')
    return {**COMMON, 'VLM': vlm}


def settings(experiment, vlm=COMMON['VLM']):
    return {**common(vlm), **VARIANTS[experiment]}


def launch_command(experiment, gpus, timestamp=None, resume=False, vlm=COMMON['VLM']):
    cmd = ['bash', str(ROOT / 'scripts/run_instance100.sh')]
    for key, value in settings(experiment, vlm).items():
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
        if name in FINGERPRINT_EXCLUDE:
            continue
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
    identity = {'experiment': args.experiment, 'settings': settings(args.experiment, args.vlm),
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
    suite_identity = {'source_digest': digest, 'python': identity['python'], 'common': common(args.vlm)}
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
    parser.add_argument('--vlm', default=COMMON['VLM'], choices=VLMS,
                        help='model for all calls; part of the run settings and the suite identity')
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
    cmd = launch_command(args.experiment, args.gpus, args.timestamp, args.resume, args.vlm)
    if args.dry_run:
        print(json.dumps({'experiment': args.experiment, 'settings': settings(args.experiment, args.vlm),
                          'command': cmd}, indent=2))
        return 0
    env = {**os.environ, **settings(args.experiment, args.vlm), 'PSGNAV_PYTHON': sys.executable,
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
