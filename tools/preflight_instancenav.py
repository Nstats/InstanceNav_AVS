#!/usr/bin/env python3
"""Check all inputs before loading detectors or spending API calls on evaluation."""
import argparse
import gzip
import importlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'habitat-lab'),str(ROOT/'GLIP')]


def check(config_path, check_api=False):
    import yaml
    from utils.model_client import get_model_client, load_environment, ModelRequestError
    load_environment()
    from configs.navigation_config import (instance_confirmation_enabled, instance_confirmation_threshold,
                                           verify_before_candidate_enabled, instance_confirmation_mode, post_confirmation_contradiction_check_enabled,
                                           preserve_positive_evidence_on_recovery_enabled,
                                           instance_collect_only_enabled,
                                           instance_description_for_exploration_enabled,
                                           relative_evidence_run_header, validate_instance_ablation)
    print('Instance ablation: ' + json.dumps(validate_instance_ablation()))
    mode = instance_confirmation_mode()
    post_check = post_confirmation_contradiction_check_enabled()
    preserve = preserve_positive_evidence_on_recovery_enabled()
    collect = instance_collect_only_enabled()
    print(f'Instance description for exploration={instance_description_for_exploration_enabled()}')
    print(f'Instance confirmation enabled={instance_confirmation_enabled()}; '
          f'threshold={instance_confirmation_threshold()}; verify_before_candidate={verify_before_candidate_enabled()}; '
          f'mode={mode}; post_confirmation_contradiction_check={post_check}; '
          f'preserve_positive_evidence_on_recovery={preserve}; instance_collect_only={collect}')
    config=yaml.safe_load(Path(config_path).read_text())
    problems=[]
    dataset=Path(config['DATASET']['DATA_PATH'].format(split=config['DATASET']['SPLIT']))
    if not dataset.exists():
        problems.append(f'Missing prepared dataset: {dataset}; run tools/prepare_instancenav.py')
    else:
        data=json.load(gzip.open(dataset,'rt'))
        scenes=sorted({ep['scene_id'] for ep in data['episodes']})
        missing=[s for s in scenes if not (Path(config['DATASET']['SCENES_DIR'])/s).is_file()]
        if missing:
            problems.append(f'Missing {len(missing)}/{len(scenes)} HM3D scene meshes; first: {missing[0]}')
        print(f"Episodes={len(data['episodes'])}; scenes={len(scenes)}; steps={config['ENVIRONMENT']['MAX_EPISODE_STEPS']}; success_distance={config['TASK']['SUCCESS']['SUCCESS_DISTANCE']}m")
    scene_config=Path(config['SIMULATOR']['SCENE_DATASET'])
    if not scene_config.is_file(): problems.append(f'Missing scene dataset config: {scene_config}')
    for path in ('GLIP/MODEL/glip_large_model.pth','data/models/sam_vit_h_4b8939.pth','data/models/groundingdino_swint_ogc.pth'):
        if not Path(path).is_file(): problems.append(f'Missing model weights: {path}')
    for name in ('habitat','habitat_sim','torch','skimage','skfmm','faiss','pytorch3d.ops', 'maskrcnn_benchmark.engine.predictor_glip'):
        try: importlib.import_module(name)
        except Exception as exc: problems.append(f'Import {name}: {type(exc).__name__}: {exc}')
    try:
        client=get_model_client()
        print('Relative evidence: ' + json.dumps(relative_evidence_run_header(
            dataset='instancenav', backend=client.backend)))
        print(f'Model backend={client.backend}; model={client.model}')
        if post_check and client.backend != 'openai':
            problems.append('Post-confirmation multi-image checking requires the openai-compatible backend')
        if check_api:
            from PIL import Image, ImageDraw
            if post_check:
                from utils.instance_verifier import verify_post_confirmation
                images = []
                for color in ('red', 'blue'):
                    image = Image.new('RGB', (128, 128), 'white')
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((24, 24, 104, 104), fill=color)
                    draw.rectangle((16, 16, 112, 112), outline='red', width=3)
                    images.append(image)
                result = verify_post_confirmation(client, images,
                                                  'square', {'intrinsic_attributes': 'The square is red.'})
                if result['decision'] != 'conflict' or result['image_index'] != 1:
                    problems.append('Unexpected multi-image structured recheck probe answer')
                else:
                    print('Qwen multi-image structured recheck probe passed')
            else:
                answer=client.complete('What solid color is this image? Return one English word.',Image.new('RGB',(32,32),'red'))
                if answer.lower().strip(' .')!='red': problems.append('Unexpected API vision probe answer')
                else: print('Qwen vision API probe passed')
    except (ValueError,ModelRequestError) as exc: problems.append(str(exc))
    for problem in problems: print('BLOCKED:',problem)
    return not problems


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/instancenav_val20_qwen.yaml')
    parser.add_argument('--check-api',action='store_true')
    args=parser.parse_args()
    raise SystemExit(0 if check(args.config,args.check_api) else 1)
