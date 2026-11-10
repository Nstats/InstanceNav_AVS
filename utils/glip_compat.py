"""Inference operators for legacy GLIP on PyTorch 2 / Blackwell.

GLIP's old THC extension cannot build on current PyTorch. Keep its parameters
and architecture, using torchvision's maintained CUDA kernels. Training-only
legacy extension entry points deliberately fail instead of returning fake values.
"""
from types import SimpleNamespace
from torchvision.ops import batched_nms, nms, deform_conv2d
from torch.nn.modules.utils import _pair


def ml_nms(boxes, scores, labels, threshold):
    # Original GLIP ml_nms uses inclusive pixel coordinates (x2-x1+1).
    adjusted=boxes.clone()
    adjusted[:,2:]+=1
    return batched_nms(adjusted, scores, labels, threshold)


def unsupported(*args, **kwargs):
    raise NotImplementedError('This legacy GLIP extension operation is unsupported in the inference runtime')


_C=SimpleNamespace(nms=nms, ml_nms=ml_nms, soft_nms=unsupported)


def deform_conv(input, offset, weight, stride=1, padding=0, dilation=1,
                groups=1, deformable_groups=1, im2col_step=64):
    return deform_conv2d(input, offset, weight, stride=_pair(stride),
                         padding=_pair(padding), dilation=_pair(dilation))


def modulated_deform_conv(input, offset, mask, weight, bias=None, stride=1,
                          padding=0, dilation=1, groups=1, deformable_groups=1):
    # GLIP DyHead passes a higher-resolution offset/mask to its coarse branch.
    # The legacy CUDA kernel indexes each batch's flat storage using the *output*
    # H,W (deform_conv_kernel_cuda.cu:607-620). Preserve those exact reads instead
    # of interpolating offsets, which would change checkpoint predictions.
    out_h = (input.shape[-2] + 2 * _pair(padding)[0] - _pair(dilation)[0] * (weight.shape[-2]-1) - 1) // _pair(stride)[0] + 1
    out_w = (input.shape[-1] + 2 * _pair(padding)[1] - _pair(dilation)[1] * (weight.shape[-1]-1) - 1) // _pair(stride)[1] + 1
    if offset.shape[-2:] != (out_h, out_w):
        offset = _legacy_spatial_layout(offset, out_h, out_w)
        mask = _legacy_spatial_layout(mask, out_h, out_w)
    return deform_conv2d(input, offset, weight, bias, stride=_pair(stride),
                         padding=_pair(padding), dilation=_pair(dilation), mask=mask)


def _legacy_spatial_layout(tensor, height, width):
    count = tensor.shape[1] * height * width
    flat = tensor.flatten(1)
    if flat.shape[1] < count:
        raise ValueError('GLIP offset tensor is smaller than the legacy kernel requires')
    return flat[:, :count].reshape(tensor.shape[0], tensor.shape[1], height, width).contiguous()
