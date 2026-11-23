# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# Optional upstream VQA client; the benchmark sets use_vqa=False.
from typing import Optional

import numpy as np

from .server_wrapper import send_request


class BLIP2Client:
    def __init__(self, port: int = 12185):
        self.url = f"http://localhost:{port}/blip2"

    def ask(self, image: np.ndarray, prompt: Optional[str] = None) -> str:
        return send_request(self.url, image=image, prompt=prompt or "")["response"]
