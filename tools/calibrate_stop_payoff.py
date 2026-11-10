#!/usr/bin/env python
"""F20 of 实验计划.md: replay payoff by stop-point distance and walking cost.

On the six k=1 runs (admission-off x3, default x3) of the verification set:
  * P(success | distance from the FIRST lock point to the goal instance), which is the payoff
    table used by analyze_relative_evidence.py (it includes later stuck losses);
  * the same for the final lock of episodes that called STOP;
  * policy steps from the final lock to the end of the episode versus the agent-to-lock
    distance at the lock, i.e. the walking cost used for reachability (about 7.2 steps / m).

Run: PYTHONPATH=habitat-lab:. $PSGNAV_PYTHON tools/calibrate_stop_payoff.py
"""
import sys, math, json, gzip
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
import numpy as np
from tools.analyze_verification_set import load_set
from tools.build_verification_set import world_to_gps
runs=['20261030_211522_2605603','20261031_133130_3729343','20261101_145827_2973537',
      '20261030_015345_3953089','20261031_204337_603781','20261101_224941_4134293']
data=load_set(Path('data/verification_set'),Path('data'),runs=runs)
ents=json.load(gzip.open('data/datasets/instancenav/val10/val10.json.gz'))['episodes']
ds={(e['scene_id'].split('/')[-1],str(e['episode_id']),str(e['goal_object_id'])):e for e in ents}
BINS=[(0,1),(1,1.5),(1.5,2),(2,3),(3,5),(5,1e9)]
def lab(d):
    for lo,hi in BINS:
        if lo<=d<hi: return f'{lo}-{hi if hi<1e9 else "inf"}'
first={};final={};walk=[]
for r in runs:
    for i,e in data[r]['episodes'].items():
        if not e['locks']: continue
        tgt=e['same_category_instances'].get(str(e['target_instance']))
        if not tgt: continue
        Ls=sorted(e['locks'],key=lambda L:L['step'])
        d0=math.dist(Ls[0]['goal_gps'],tgt['gps'])
        first.setdefault(lab(d0),[]).append(bool(e['success']))
        if e['termination']=='stop_called':
            dn=math.dist(Ls[-1]['goal_gps'],tgt['gps'])
            final.setdefault(lab(dn),[]).append(bool(e['success']))
            # steps from final lock to the stop, vs agent-to-lock distance
            de=ds[tuple(e['key'])]
            fr={f['step']:f for f in e['frames']}
            f=fr.get(Ls[-1]['step'])
            if f:
                a=world_to_gps(f['position'],de['start_position'],de['start_rotation'])
                walk.append((math.dist(a,Ls[-1]['goal_gps']), e['frames'][-1]['step']-Ls[-1]['step'], e.get('num_steps')))
print('P(success | FIRST lock distance to target)')
for lo,hi in BINS:
    k=f'{lo}-{hi if hi<1e9 else "inf"}'; v=first.get(k,[]); print(f'  {k:8s} n={len(v):4d} succ={np.mean(v) if v else float("nan"):.3f}')
print('P(success | FINAL lock distance, stop_called only)')
for lo,hi in BINS:
    k=f'{lo}-{hi if hi<1e9 else "inf"}'; v=final.get(k,[]); print(f'  {k:8s} n={len(v):4d} succ={np.mean(v) if v else float("nan"):.3f}')
w=np.array(walk,float)
print('walk samples',len(w),'dist p50/p90',np.percentile(w[:,0],[50,90]).round(2),'steps p50/p90',np.percentile(w[:,1],[50,90]))
for lo,hi in ((0,1.5),(1.5,2.5),(2.5,4),(4,99)):
    m=(w[:,0]>=lo)&(w[:,0]<hi)
    if m.sum(): print(f'  d in [{lo},{hi}): n={m.sum()} steps p50={np.median(w[m,1]):.0f} p90={np.percentile(w[m,1],90):.0f} steps/m p50={np.median(w[m,1]/np.maximum(w[m,0],0.5)):.1f}')
A=np.vstack([w[:,0],np.ones(len(w))]).T
print('fit steps = b*d + a :',np.linalg.lstsq(A,w[:,1],rcond=None)[0].round(2))
g=[e['info']['geodesic_distance']/max(e['info']['euclidean_distance'],.1) for e in ents]
print('geodesic/euclid ratio p50/p90',np.percentile(g,[50,90]).round(2))
