# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import base64
import fcntl
import os
import random
import socket
import time
from typing import Any, Dict

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request


class ServerMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def process_payload(self, payload: dict) -> dict:
        raise NotImplementedError


def host_model(model: Any, name: str, port: int = 5000) -> None:
    """
    Hosts a model as a REST API using Flask.
    """
    app = Flask(__name__)

    @app.after_request
    def identify_process(response):
        response.headers['X-VLFM-Pid'] = str(os.getpid())
        return response

    @app.route(f"/{name}", methods=["POST"])
    def process_request() -> Dict[str, Any]:
        payload = request.json
        return jsonify(model.process_payload(payload))

    app.run(host="localhost", port=port)


def bool_arr_to_str(arr: np.ndarray) -> str:
    """Converts a boolean array to a string."""
    packed_str = base64.b64encode(arr.tobytes()).decode()
    return packed_str


def str_to_bool_arr(s: str, shape: tuple) -> np.ndarray:
    """Converts a string to a boolean array."""
    # Convert the string back into bytes using base64 decoding
    bytes_ = base64.b64decode(s)

    # Convert bytes to np.uint8 array
    bytes_array = np.frombuffer(bytes_, dtype=np.uint8)

    # Reshape the data back into a boolean array
    unpacked = bytes_array.reshape(shape)
    return unpacked


def image_to_str(img_np: np.ndarray, quality: int = 90) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    retval, buffer = cv2.imencode(".jpg", img_np, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


def str_to_image(img_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
    return img_np


def send_request(url: str, **kwargs: Any) -> dict:
    response = {}
    for attempt in range(10):
        try:
            response = _send_request(url, **kwargs)
            break
        except Exception as e:
            if attempt == 9:
                raise RuntimeError(f'VLFM model request failed: {url}') from e
            else:
                print(f"Error: {e}. Retrying in 20-30 seconds...")
                time.sleep(20 + random.random() * 10)

    return response


def _send_request(url: str, **kwargs: Any) -> dict:
    lockfiles_dir = "lockfiles"
    os.makedirs(lockfiles_dir, exist_ok=True)
    filename = url.replace("/", "_").replace(":", "_") + ".flock"
    filename = filename.replace("localhost", socket.gethostname())
    # Keep the inode: unlinking it could let a newcomer bypass existing waiters.
    # Closing the descriptor (including on process exit) releases the lock.
    with open(os.path.join(lockfiles_dir, filename), "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

        # Create a payload dict which is a clone of kwargs but all np.array values are
        # converted to strings
        payload = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                payload[k] = image_to_str(v, quality=kwargs.get("quality", 90))
            else:
                payload[k] = v

        # Set the headers
        headers = {"Content-Type": "application/json"}

        start_time = time.time()
        while True:
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=1)
                if resp.status_code == 200:
                    return resp.json()
                else:
                    raise Exception("Request failed")
            except (
                requests.exceptions.Timeout,
                requests.exceptions.RequestException,
            ) as e:
                print(e)
                if time.time() - start_time > 20:
                    raise Exception("Request timed out after 20 seconds")
