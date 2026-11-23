"""Depth hole filling for VLFM's normalized [0,1] observations, not PSG-Nav.

The upstream IP-Basic multiscale procedure bins at 15 and 30 metres: on these
normalized inputs only the nearest bin is ever populated. This narrow implementation
matches filter_depth(..., blur_type=None) on that domain and keeps all observed pixels.
"""
import cv2
import numpy as np


def fill_normalized_depth(depth):
    near = depth > 0.1
    inverse = np.where(near, 100 - depth, depth)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
    spread = cv2.dilate(np.where(near, inverse, 0), cross)
    surface = np.where(spread > 0.1, spread, inverse)
    square = np.ones((5, 5), dtype=np.uint8)
    surface = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, square)
    surface = np.where(surface > 0.1, cv2.medianBlur(surface, 5), surface)

    rows = np.arange(depth.shape[0])[:, None]
    below = rows >= np.argmax(surface > 0.1, axis=0)
    surface = np.where(below & (surface <= 0.1), cv2.dilate(surface, np.ones((9, 9), np.uint8)), surface)
    below = rows >= np.argmax(surface > 0.1, axis=0)
    for _ in range(6):
        surface = np.where(below & (surface < 0.1), cv2.dilate(surface, square), surface)
    surface = np.where(below & (surface > 0.1), cv2.medianBlur(surface, 5), surface)
    return np.where(depth != 0, depth, np.where(surface > 0.1, 100 - surface, surface))
