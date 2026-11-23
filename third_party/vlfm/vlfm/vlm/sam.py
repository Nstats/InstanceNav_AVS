# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, List, Optional

import numpy as np
import torch

from .server_wrapper import send_request, str_to_bool_arr

class MobileSAM:
    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_t",
        device: Optional[Any] = None,
    ) -> None:
        from mobile_sam import SamPredictor, sam_model_registry

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.device = device

        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()
        self.predictor = SamPredictor(mobile_sam)

    def segment_bbox(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Segments the object in the given bounding box from the image.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            bbox (List[int]): The bounding box as a numpy array in the
                format [x1, y1, x2, y2].

        Returns:
            np.ndarray: The segmented object as a numpy array (boolean mask). The mask
                is the same size as the bbox, cropped out of the image.

        """
        with torch.inference_mode():
            self.predictor.set_image(image)
            masks, _, _ = self.predictor.predict(box=np.array(bbox), multimask_output=False)

        return masks[0]


class MobileSAMClient:
    def __init__(self, port: int = 12183):
        self.url = f"http://localhost:{port}/mobile_sam"

    def segment_bbox(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        response = send_request(self.url, image=image, bbox=bbox)
        cropped_mask_str = response["cropped_mask"]
        cropped_mask = str_to_bool_arr(cropped_mask_str, shape=tuple(image.shape[:2]))

        return cropped_mask
