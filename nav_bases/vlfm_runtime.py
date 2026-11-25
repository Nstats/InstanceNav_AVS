"""Repository-local VLFM paths. Import/activate only from the VLFM entry points."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'third_party' / 'vlfm'
UPSTREAM_REVISION = '584ed56008754fde7997d904983607def8328322'
POINTNAV = 'pointnav_weights_state_dict.pth'
BERT = ('bert-base-uncased/config.json', 'bert-base-uncased/vocab.txt', 'bert-base-uncased/pytorch_model.bin')
ASSETS = {
    'navigation': (POINTNAV,),
    'grounding_dino': ('groundingdino_swint_ogc.pth', *BERT),
    'blip2itm': ('blip2_pretrained.pth', 'eva_vit_g.pth', *BERT),
    'sam': ('mobile_sam.pt',),
    'yolov7': ('yolov7-e6e.pt',),
}
# YOLOv7's official checkpoint contains Python classes, not just tensors. Only this
# exact upstream artifact may enter its pickle loader in the isolated YOLO process.
YOLO_SHA256 = 'b370120a414bf32b5d65fc808e5a32c8d9b3c63902d1bc41894fc9d86eccf9cb'


def asset_dir():
    return Path(os.environ.get('VLFM_ASSET_DIR', ROOT / 'data/vlfm')).expanduser().resolve()


def asset_path(name):
    if name == POINTNAV and os.environ.get('VLFM_POINTNAV_WEIGHTS'):
        return Path(os.environ['VLFM_POINTNAV_WEIGHTS']).expanduser().resolve()
    return asset_dir() / name


def check_assets(groups=ASSETS):
    paths = [asset_path(name) for name in dict.fromkeys(n for group in groups for n in ASSETS[group])]
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError('Missing VLFM assets (copy the actual files, not broken symlinks):\n  '
                                + '\n  '.join(missing) + '\nSee docs/VLFM_H20.md.')
    return paths


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_avs_settings(dataset='instancenav'):
    """Check the effective switches, not the preset name. No model or GPU imports."""
    from configs.navigation_config import (
        instance_collect_only_enabled, instance_confirmation_enabled, instance_confirmation_mode,
        instance_confirmation_threshold, post_confirmation_contradiction_check_enabled,
        preserve_positive_evidence_on_recovery_enabled, relative_evidence_enabled,
        validate_instance_ablation, verify_before_candidate_enabled)

    summary = validate_instance_ablation()
    enabled = instance_confirmation_enabled()
    requirements = {'RAG_MODE=0': os.environ.get('RAG_MODE', '0') == '0',
                    'ENABLE_RELATIVE_EVIDENCE=false': not relative_evidence_enabled(),
                    'INSTANCE_COLLECT_ONLY=false': not instance_collect_only_enabled()}
    threshold, admission = instance_confirmation_threshold(), verify_before_candidate_enabled()
    if enabled:
        approach = summary['candidate_approach']
        requirements.update({
            'INSTANCE_CONFIRMATION_MODE=positive_match': instance_confirmation_mode() == 'positive_match',
            'INSTANCE_REJECTION_POLICY=none': summary['rejection_policy'] == 'none',
            'ENABLE_POST_CONFIRMATION_CONTRADICTION_CHECK=false': not post_confirmation_contradiction_check_enabled(),
            'PRESERVE_POSITIVE_EVIDENCE_ON_RECOVERY=false': not preserve_positive_evidence_on_recovery_enabled(),
        })
        if approach:
            requirements.update({
                'InstanceNav for candidate approach': dataset == 'instancenav',
                'INSTANCE_VERIFICATION_SCOPE=instance': summary['scope'] == 'instance',
                'INSTANCE_CONFIRMATION_THRESHOLD=1 or 2': threshold in (1, 2),
                'one confirmation for RA': not admission or threshold == 1,
                'INSTANCE_CONFIRMATION_COUNT_UNIT=frame for E5': threshold == 1 or summary['count_unit'] == 'frame',
            })
        else:
            requirements.update({'INSTANCE_CONFIRMATION_THRESHOLD=1': threshold == 1,
                                 'VERIFY_BEFORE_CANDIDATE=false': not admission})
    problems = [rule for rule, ok in requirements.items() if not ok]
    if problems:
        raise ValueError('NAV_BASE=vlfm requires ' + '; '.join(problems))
    return dict(summary, avs=enabled, threshold=threshold, admission=admission)


def activate():
    """Make only bundled packages visible. No conda activation, sibling paths or .pth files."""
    path = str(SOURCE)
    if path not in sys.path:
        sys.path.insert(0, path)
    import vlfm
    if Path(vlfm.__file__).resolve().parent != SOURCE / 'vlfm':
        raise ImportError(f'Expected the bundled VLFM at {SOURCE}, got {vlfm.__file__}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check VLFM model files; never start a service or navigate.')
    parser.add_argument('command', choices=['check'])
    parser.add_argument('--group', choices=ASSETS, action='append', help='only check one component (repeatable)')
    args = parser.parse_args()
    try:
        for file in check_assets(args.group or ASSETS):
            print(f'OK {file}')
    except FileNotFoundError as error:
        parser.exit(1, str(error) + '\n')
