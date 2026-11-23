# Copyright (c) 2022-2023, salesforce.com, inc. All rights reserved.
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause AND MIT
# Adapted from LAVIS and VLFM; see LICENSE and SOURCES.
"""Inference for exactly blip2_image_text_matching/pretrain, match_head='itc'."""

import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from transformers import BertConfig, BertTokenizer

from .eva_vit import WIDTH, EVAViTG
from .qformer import QFormer


_eval_image = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


def _eval_caption(text):
    text = re.sub(r'([.!"()*#:;~])', ' ', text.lower())
    text = re.sub(r'\s{2,}', ' ', text).rstrip('\n').strip(' ')
    return ' '.join(text.split(' ')[:50])


def _restore(module, path, *, blip=False):
    # Both canonical LAVIS files are torch zip archives. Map storage lazily on
    # CPU. Never change the process's deserialization policy or retry as pickle.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["model"] if blip else checkpoint
    missing, _ = module.load_state_dict(state, strict=False)
    # Ignore extra classifier/training keys, but never score with random weights.
    # The pretrain checkpoint intentionally omits the already restored frozen EVA.
    if blip:
        missing = [key for key in missing if not key.startswith("visual_encoder.")]
    if missing:
        raise RuntimeError(f"{path.name} is missing inference weights: {', '.join(missing)}")


class BLIP2ITM(nn.Module):
    """VLFM's maximum image-query/text cosine, using trusted local assets only."""

    def __init__(self, asset_dir: Path, device=None):
        super().__init__()
        root = Path(asset_dir)
        eva = root / "eva_vit_g.pth"
        blip = root / "blip2_pretrained.pth"
        bert = root / "bert-base-uncased"
        files = (eva, blip, bert / "config.json", bert / "vocab.txt", bert / "pytorch_model.bin")
        missing = [str(f.relative_to(root)) for f in files if not f.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing local BLIP2 assets under {root}: {', '.join(missing)}")

        device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = BertTokenizer.from_pretrained(bert, local_files_only=True)
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        config = BertConfig.from_json_file(str(bert / "config.json"))

        # Skip a gigaparameter random initialization. _restore checks that every
        # EVA tensor was supplied. LAVIS converts only Conv/Linear to fp16, not
        # the class/position embeddings, Q/V biases, or layer norms.
        with torch.device("meta"):
            self.visual_encoder = EVAViTG()
            for layer in self.visual_encoder.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    layer.half()
        self.visual_encoder.to_empty(device="cpu")
        _restore(self.visual_encoder, eva)

        self.Qformer = QFormer(config, len(self.tokenizer), WIDTH, bert / "pytorch_model.bin")
        self.query_tokens = nn.Parameter(torch.empty(1, 32, config.hidden_size))
        self.ln_vision = nn.LayerNorm(WIDTH)
        self.vision_proj = nn.Linear(config.hidden_size, 256)
        self.text_proj = nn.Linear(config.hidden_size, 256)
        _restore(self, blip, blip=True)
        self.requires_grad_(False).eval()
        if device.type == "cpu":
            self.float()  # LAVIS load_model_and_preprocess on CPU
        self.to(device)

    @torch.inference_mode()
    def cosine(self, image_np: np.ndarray, text: str) -> float:
        """Score one HWC RGB uint8 frame and a caption. No temperature or softmax."""
        device = self.query_tokens.device
        image = _eval_image(Image.fromarray(image_np)).unsqueeze(0).to(device)
        tokens = self.tokenizer(_eval_caption(text), truncation=True, max_length=32, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16) if image.is_cuda else nullcontext():
            x = self.visual_encoder(image)
            # LAVIS's LayerNorm runs in float and returns its input dtype.
            x = self.ln_vision(x.float()).to(x.dtype)
        queries = self.Qformer.encode_queries(self.query_tokens, x.float())
        cls = self.Qformer.encode_text(tokens.input_ids, tokens.attention_mask)[:, 0, :]
        image_feats = F.normalize(self.vision_proj(queries), dim=-1)
        text_feat = F.normalize(self.text_proj(cls), dim=-1)
        return torch.bmm(image_feats, text_feat.unsqueeze(-1)).max(dim=1).values.item()
