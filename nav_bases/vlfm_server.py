"""One offline model per process. No YOLO `models`/`utils` or server paths enter PSG-Nav."""
import argparse
import os
import sys
from types import SimpleNamespace

from . import vlfm_runtime as runtime


def build_handler(name):
    """Construct the upstream model and its JSON protocol handler (no port opened here)."""
    runtime.check_assets([name])
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    runtime.activate()
    from vlfm.vlm.server_wrapper import bool_arr_to_str, str_to_image

    assets = runtime.asset_dir()
    if name == 'grounding_dino':
        import torch
        from vlfm.vlm.grounding_dino import GroundingDINO
        model = GroundingDINO(assets / 'bert-base-uncased',
                              runtime.ROOT / 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                              assets / 'groundingdino_swint_ogc.pth',
                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if model.device.type == 'cuda':
            # Each service has its own process; leave cuDNN and the other models alone.
            torch.backends.cuda.matmul.allow_tf32 = True
            print('[GroundingDINO] CUDA matmul allow_tf32=True', flush=True)
        return 'gdino', lambda p: model.predict(str_to_image(p['image']), caption=p['caption']).to_json()
    if name == 'blip2itm':
        from lavis_itm import BLIP2ITM
        model = BLIP2ITM(assets)
        return 'blip2itm', lambda p: {'response': model.cosine(str_to_image(p['image']), p['txt'])}
    if name == 'sam':
        from vlfm.vlm.sam import MobileSAM
        model = MobileSAM(assets / 'mobile_sam.pt')
        return 'mobile_sam', lambda p: {'cropped_mask': bool_arr_to_str(
            model.segment_bbox(str_to_image(p['image']), p['bbox']))}
    if name == 'yolov7':
        # The trusted release pickle imports `models.yolo` and `utils.*` by name.
        # This directory is added ONLY in this dedicated subprocess.
        weights = assets / 'yolov7-e6e.pt'
        if runtime.sha256(weights) != runtime.YOLO_SHA256:
            raise ValueError(f'Not the verified official YOLOv7-E6E checkpoint: {weights}')
        sys.path.insert(0, str(runtime.SOURCE / 'yolov7'))
        from vlfm.vlm.yolov7 import YOLOv7
        model = YOLOv7(str(weights))
        return 'yolov7', lambda p: model.predict(str_to_image(p['image'])).to_json()
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=tuple(n for n in runtime.ASSETS if n != 'navigation'))
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args()
    endpoint, handle = build_handler(args.model)
    from vlfm.vlm.server_wrapper import host_model
    host_model(SimpleNamespace(process_payload=handle), name=endpoint, port=args.port)


if __name__ == '__main__':
    main()
