"""Offline coverage for the VLFM scorer. Real-weight regression is opt-in, CPU only."""

import os
import socket
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def itm(monkeypatch):
    def offline(*args, **kwargs):
        pytest.fail("BLIP2 inference must not use the network")

    monkeypatch.setattr(socket, "getaddrinfo", offline)
    monkeypatch.setattr(socket.socket, "connect", offline)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "third_party" / "vlfm"))
    from lavis_itm import model
    return model


def test_all_missing_assets_are_named_without_cache_fallback(itm, tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        itm.BLIP2ITM(tmp_path, device="cpu")
    for name in ("eva_vit_g.pth", "blip2_pretrained.pth", "bert-base-uncased/config.json",
                 "bert-base-uncased/vocab.txt", "bert-base-uncased/pytorch_model.bin"):
        assert name in str(exc.value)


def test_lavis_caption_processor(itm):
    assert itm._eval_caption('  RED!  "chair": blue-green, (mat) #42?  \n') == "red chair blue-green, mat 42?"
    words = [str(i) for i in range(55)]
    assert itm._eval_caption(" ".join(words)) == " ".join(words[:50])


def test_lavis_eval_resizes_without_crop_and_normalizes_rgb(itm):
    from PIL import Image

    rgb = np.arange(3 * 7 * 11, dtype=np.uint8).reshape(7, 11, 3)
    pil = Image.fromarray(rgb)
    expected = np.asarray(pil.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    expected -= np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    expected /= np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    np.testing.assert_allclose(itm._eval_image(pil).numpy(), expected.transpose(2, 0, 1), rtol=0, atol=1e-6)


def test_incomplete_checkpoint_cannot_leave_random_weights(itm, tmp_path):
    import torch

    layer = torch.nn.Linear(3, 4)
    path = tmp_path / "partial.pth"
    torch.save({"weight": layer.weight.detach(), "head.unused": torch.ones(1)}, path)
    with pytest.raises(RuntimeError, match="partial.pth is missing inference weights: bias"):
        itm._restore(layer, path)


@pytest.mark.skipif(not os.environ.get("VLFM_BLIP_TEST_ASSET_DIR"), reason="requires original local BLIP2 weights")
def test_official_pretrain_itc_score_cpu(itm):
    # Golden from the unmodified VLFM BLIP2ITM.cosine / salesforce-lavis 1.0.2.
    y, x = np.indices((80, 120))
    image = np.stack((2 * x, 3 * y, x + y), axis=-1).astype(np.uint8)
    scorer = itm.BLIP2ITM(Path(os.environ["VLFM_BLIP_TEST_ASSET_DIR"]), device="cpu")
    score = scorer.cosine(image, "  A red, GREEN and blue gradient!  ")
    assert isinstance(score, float)
    assert score == pytest.approx(0.4260253608226776, abs=1e-5)
