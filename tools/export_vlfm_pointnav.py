#!/usr/bin/env python3
"""Export VLFM's PointNav checkpoint to a plain state_dict the VLFM base can load.

The official data/pointnav_weights.pth pickles a habitat-lab 0.2.4 OmegaConf config,
so unpickling it needs habitat 0.2.4. The navigation process runs the bundled
habitat-lab 0.2.1, so this tool runs once with VLFM's vendored dependencies:

    PYTHONPATH=/data/lxj/vlfm/.vlfm-deps:/data/lxj/vlfm python tools/export_vlfm_pointnav.py

It writes only tensors (loadable with weights_only=True) plus a JSON note with both
SHA-256 digests. With --check it also compares the exported network
(nav_bases/vlfm_pointnav.py) with habitat_baselines' PointNavResNetPolicy on random
inputs; the actions must be identical.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path('/data/lxj/vlfm/data/pointnav_weights.pth')
DEFAULT_TARGET = ROOT / 'data/vlfm/pointnav_weights_state_dict.pth'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(source, target):
    ckpt = torch.load(source, map_location='cpu', weights_only=False)
    state = {k: v.clone() for k, v in ckpt['state_dict'].items()}
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, target)
    note = {'source': str(source), 'source_sha256': sha256(source),
            'target_sha256': sha256(target), 'keys': len(state)}
    target.with_suffix('.json').write_text(json.dumps(note, indent=2) + '\n')
    print(json.dumps(note, indent=2))
    return ckpt


def check(ckpt, target, trials=20):
    """Exported discrete network == habitat_baselines policy, step by step."""
    import numpy as np
    from gym import spaces
    from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
    import sys
    sys.path.insert(0, str(ROOT))
    from nav_bases.vlfm_pointnav import load_discrete_pointnav_policy

    obs_space = spaces.Dict({
        'depth': spaces.Box(low=0.0, high=1.0, shape=(224, 224, 1), dtype=np.float32),
        'pointgoal_with_gps_compass': spaces.Box(low=np.finfo(np.float32).min,
                                                 high=np.finfo(np.float32).max,
                                                 shape=(2,), dtype=np.float32)})
    reference = PointNavResNetPolicy.from_config(ckpt['config'], obs_space, spaces.Discrete(4))
    reference.load_state_dict(ckpt['state_dict'])
    ours = load_discrete_pointnav_policy(str(target))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reference.to(device).eval()
    ours.to(device).eval()
    gen = torch.Generator().manual_seed(0)
    h_ref = torch.zeros(1, reference.net.num_recurrent_layers, 512, device=device)
    h_our = h_ref.clone()
    prev_ref = torch.zeros(1, 1, dtype=torch.long, device=device)
    prev_our = prev_ref.clone()
    worst = 0.0
    with torch.no_grad():
        for step in range(trials):
            obs = {'depth': torch.rand(1, 224, 224, 1, generator=gen).to(device),
                   'pointgoal_with_gps_compass': (torch.rand(1, 2, generator=gen) * 4 - 1).to(device)}
            masks = torch.tensor([[step != 0]], dtype=torch.bool, device=device)
            out = reference.act(obs, h_ref, prev_ref, masks, deterministic=True)
            a_ref, h_ref = out.actions, out.rnn_hidden_states
            a_our, h_our = ours.act(obs, h_our, prev_our, masks.view(-1), deterministic=True)
            if not torch.equal(a_ref.long(), a_our.long()):
                raise SystemExit(f'Action mismatch at step {step}: {a_ref.tolist()} vs {a_our.tolist()}')
            worst = max(worst, float((h_ref - h_our).abs().max()))
            prev_ref, prev_our = a_ref.long(), a_our.long()
    print(f'CHECK_OK steps={trials} max_hidden_abs_diff={worst:.3e}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--target', type=Path, default=DEFAULT_TARGET)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    ckpt = export(args.source, args.target)
    if args.check:
        check(ckpt, args.target)


if __name__ == '__main__':
    main()
