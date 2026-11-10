"""Bounded exact-input caches for fixed, evaluation-only perception models.

No approximate matching or frame skipping: callers still run all mapping and
navigation updates. Returned values are cloned because downstream code mutates
labels, language features, and predictor state.
"""
from collections import OrderedDict
import copy
from functools import lru_cache
import hashlib
import json

import numpy as np
import torch


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return type(value)((key, clone_tree(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    return copy.deepcopy(value)


def input_key(value):
    if isinstance(value, np.ma.MaskedArray):
        return ('masked', input_key(value.data), input_key(np.ma.getmaskarray(value)))
    if isinstance(value, np.ndarray):
        return ('array', str(value.dtype), value.shape, hashlib.sha256(value.tobytes()).digest())
    if isinstance(value, torch.Tensor):
        array = value.detach().contiguous().cpu().numpy()
        return ('tensor', str(value.device), str(array.dtype), array.shape,
                hashlib.sha256(array.tobytes()).digest())
    if isinstance(value, dict):
        return tuple((key, input_key(item)) for key, item in sorted(value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(input_key(item) for item in value)
    return value


_distance_cache = None


def exact_fmm_distance(field, dx=1):
    """Same solver, reused only for byte-identical field, mask and spacing."""
    import skfmm
    global _distance_cache
    if _distance_cache is None:
        _distance_cache = ExactCache(4)
    key = (input_key(field), dx)
    value = _distance_cache.get(key)
    if value is None:
        value = skfmm.distance(field, dx=dx)
        _distance_cache.put(key, value)
    return value


class ExactCache:
    def __init__(self, size):
        self.size = size
        self.data = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.data:
            self.hits += 1
            self.data.move_to_end(key)
            return clone_tree(self.data[key])
        self.misses += 1
        return None

    def put(self, key, value):
        self.data[key] = clone_tree(value)
        self.data.move_to_end(key)
        while len(self.data) > self.size:
            self.data.popitem(last=False)


def cache_eval_module(module, size=4):
    """Only for frozen models; never reuse outputs in training/grad-enabled mode."""
    cache = ExactCache(size)
    forward = module.forward

    def cached(*args, **kwargs):
        if module.training or torch.is_grad_enabled():
            cache.data.clear()
            return forward(*args, **kwargs)
        key = (torch.backends.cuda.matmul.allow_tf32, input_key((args, kwargs)))
        output = cache.get(key)
        if output is None:
            output = forward(*args, **kwargs)
            cache.put(key, output)
        return output

    module.forward = cached
    return cache


def cache_glip(demo, size=4):
    cache = ExactCache(size)
    inference = demo.inference
    attributes = ('entities', 'plus', 'positive_map_label_to_token')

    def cached(image, caption):
        if demo.model.training:
            cache.data.clear()
            return inference(image, caption)
        key = (torch.backends.cuda.matmul.allow_tf32, image.shape, str(image.dtype), hashlib.sha256(image.tobytes()).digest(),
               json.dumps(caption, sort_keys=True), repr(demo.confidence_threshold))
        result = cache.get(key)
        if result is not None:
            prediction, state = result
            for name, value in state.items():
                setattr(demo, name, value)
            return prediction
        prediction = inference(image, caption)
        cache.put(key, (prediction, {name: getattr(demo, name) for name in attributes if hasattr(demo, name)}))
        return prediction

    demo.inference = cached
    return cache


def cache_sam_images(predictor, size=4):
    cache = ExactCache(size)
    set_image = predictor.set_image
    attributes = ('features', 'original_size', 'input_size', 'is_image_set')

    def cached(image, image_format='RGB'):
        if predictor.model.training:
            cache.data.clear()
            return set_image(image, image_format=image_format)
        key = (torch.backends.cuda.matmul.allow_tf32, image.shape, str(image.dtype), image_format, hashlib.sha256(image.tobytes()).digest())
        state = cache.get(key)
        if state is None:
            result = set_image(image, image_format=image_format)
            cache.put(key, {name: getattr(predictor, name) for name in attributes})
            return result
        for name, value in state.items():
            setattr(predictor, name, value)

    predictor.set_image = cached
    return cache


def install_exact_caches(agent):
    caches = {}
    if agent.glip_demo is not None:
        caches['glip_images'] = cache_glip(agent.glip_demo)
        caches['glip_text'] = cache_eval_module(agent.glip_demo.model.language_backbone)
        # Activation checkpointing only trades compute for saved backward state.
        # Evaluation has no backward pass and uses the identical blocks directly.
        for module in agent.glip_demo.model.modules():
            if hasattr(module, 'use_checkpoint'):
                module.use_checkpoint = False
    if agent.scenegraph.sam_variant == 'groundedsam':
        dino, sam = agent.scenegraph.mask_generator
        caches['dino_text'] = cache_eval_module(dino.bert)
        caches['sam_images'] = cache_sam_images(sam)
        install_sam_position_cache()
    agent.inference_caches = caches


def install_sam_position_cache():
    # Original SAM rebuilds CPU indices and indexes CUDA parameters 64 times
    # per image. These positions depend only on fixed weights and spatial size.
    from segment_anything.segment_anything.modeling import image_encoder
    original = image_encoder.get_rel_pos
    if getattr(original, '_psgnav_cached', False):
        return

    @lru_cache(maxsize=128)
    def positions(q_size, k_size, parameter, version):
        return original(q_size, k_size, parameter).detach()

    def cached(q_size, k_size, parameter):
        if torch.is_grad_enabled():
            return original(q_size, k_size, parameter)
        return positions(q_size, k_size, parameter, parameter._version).clone()

    cached._psgnav_cached = True
    image_encoder.get_rel_pos = cached
