# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as F

from vlfm.vlm.detections import ObjectDetections

from .server_wrapper import send_request
CLASSES = "chair . person . dog ."  # Default classes. Can be overridden at inference.


class GroundingDINO:
    def __init__(
        self,
        bert_dir: str,
        config_path: str,
        weights_path: str,
        caption: str = CLASSES,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: torch.device = torch.device("cuda"),
    ):
        from groundingdino.models import build_model
        from groundingdino.util.misc import clean_state_dict
        from groundingdino.util.slconfig import SLConfig

        # Same loader as groundingdino.util.inference.load_model, but explicitly
        # point our checkout's BERT setting at local files before model construction.
        config = SLConfig.fromfile(str(config_path))
        config.device = str(device)
        config.bert_base_uncased_path = str(bert_dir)
        model = build_model(config)
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(clean_state_dict(state['model']), strict=False)
        self.device = device
        self.model = model.eval().to(device)
        self.caption = caption
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def predict(self, image: np.ndarray, caption: Optional[str] = None) -> ObjectDetections:
        """
        This function makes predictions on an input image tensor or numpy array using a
        pretrained model.

        Arguments:
            image (np.ndarray): An image in the form of a numpy array.
            caption (Optional[str]): A string containing the possible classes
                separated by periods. If not provided, the default classes will be used.

        Returns:
            ObjectDetections: An instance of the ObjectDetections class containing the
                object detections.
        """
        from groundingdino.util.inference import predict

        # Convert image to tensor and normalize from 0-255 to 0-1
        image_tensor = F.to_tensor(image)
        image_transformed = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if caption is None:
            caption_to_use = self.caption
        else:
            caption_to_use = caption
        print("Caption:", caption_to_use)
        with torch.inference_mode():
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_transformed,
                caption=caption_to_use,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=str(self.device),
            )
        detections = ObjectDetections(boxes, logits, phrases, image_source=image)

        # Remove detections whose class names do not exactly match the provided classes
        classes = caption_to_use[: -len(" .")].split(" . ")
        detections.filter_by_class(classes)

        return detections


class GroundingDINOClient:
    def __init__(self, port: int = 12181):
        self.url = f"http://localhost:{port}/gdino"

    def predict(self, image_numpy: np.ndarray, caption: Optional[str] = "") -> ObjectDetections:
        response = send_request(self.url, image=image_numpy, caption=caption)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections
