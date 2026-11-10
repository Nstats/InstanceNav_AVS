#!/usr/bin/env python3
"""Load the original detector checkpoints and run one real image through each."""
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'habitat-lab'),str(ROOT/'GLIP')]
os.chdir(ROOT)
os.environ.setdefault('NLTK_DATA',str(ROOT/'data/nltk_data'))
import torch
import numpy as np
from PIL import Image


def main():
    from GLIP.maskrcnn_benchmark.config import cfg
    from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
    cfg.merge_from_file('GLIP/configs/pretrain/glip_Swin_L.yaml')
    cfg.merge_from_list(['MODEL.WEIGHT','GLIP/MODEL/glip_large_model.pth','MODEL.DEVICE','cuda'])
    print('Loading GLIP',flush=True)
    model=GLIPDemo(cfg,min_image_size=800,confidence_threshold=0.61,show_mask_heatmaps=False)
    image=Image.open('assets/demo1.gif').convert('RGB');image.thumbnail((640,480))
    rgb=np.array(image)
    started=time.monotonic()
    with torch.inference_mode():
        result=model.inference(rgb[:,:,::-1].copy(),'chair. table. sofa. bed. plant. tv. toilet.')
    print('GLIP_SMOKE_OK',len(result),'boxes',round(time.monotonic()-started,2),'seconds',flush=True)
    del model
    torch.cuda.empty_cache()
    from utils.utils_sg_uq.sam_utils import load_model, get_grounding_output
    from GroundingDINO.groundingdino.datasets import transforms as T
    transform=T.Compose([T.RandomResize([800],max_size=1333),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    tensor,_=transform(image,None)
    dino=load_model('GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py','data/models/groundingdino_swint_ogc.pth',None,'cuda')
    with torch.inference_mode():
        boxes,labels=get_grounding_output(dino,tensor,'chair. table. sofa.',.3,.25,device='cuda')
    print('DINO_SMOKE_OK',len(boxes),'boxes',flush=True)
    del dino
    torch.cuda.empty_cache()
    from segment_anything.segment_anything import sam_model_registry,SamPredictor
    predictor=SamPredictor(sam_model_registry['vit_h'](checkpoint='data/models/sam_vit_h_4b8939.pth').to('cuda'))
    with torch.inference_mode():
        predictor.set_image(rgb)
        masks,scores,_=predictor.predict(box=np.array([0,0,rgb.shape[1]-1,rgb.shape[0]-1]),multimask_output=False)
    assert masks.shape[1:]==rgb.shape[:2]
    print('SAM_SMOKE_OK',masks.shape,flush=True)


if __name__=='__main__':main()
