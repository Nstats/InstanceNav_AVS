# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# The offline inference implementation is in lavis_itm; navigation needs only the client.
import numpy as np

from .server_wrapper import send_request


class BLIP2ITMClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        print(f"BLIP2ITMClient.cosine: {image.shape}, {txt}")
        response = send_request(self.url, image=image, txt=txt)
        return float(response["response"])
