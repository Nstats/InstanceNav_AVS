from types import SimpleNamespace
import numpy as np
import torch

from utils.inference_cache import cache_eval_module, cache_glip, cache_sam_images


def test_fmm_cache_preserves_masks_and_returns_copies():
    import skfmm
    from utils.inference_cache import exact_fmm_distance
    field = np.ones((12, 12))
    field[2, 2] = 0
    masked = np.ma.masked_array(field, mask=np.zeros_like(field, dtype=bool))
    original = skfmm.distance(masked, dx=1)
    first = exact_fmm_distance(masked)
    np.testing.assert_array_equal(first, original)
    first[:] = 999
    np.testing.assert_array_equal(exact_fmm_distance(masked), original)
    masked.mask[4:8, 4] = True
    expected = skfmm.distance(masked, dx=1)
    actual = exact_fmm_distance(masked)
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.mask, expected.mask)


def test_text_cache_clones_mutable_results_and_bypasses_training():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, inputs):
            self.calls += 1
            return {'hidden': inputs['ids'].float() * 2}
    model = Encoder().eval()
    cache = cache_eval_module(model, size=1)
    with torch.no_grad():
        first = model({'ids': torch.tensor([1, 2])})
        first['hidden'].zero_()
        assert torch.equal(model({'ids': torch.tensor([1, 2])})['hidden'], torch.tensor([2., 4.]))
        assert model.calls == 1 and cache.hits == 1
        model({'ids': torch.tensor([3])})
        model({'ids': torch.tensor([1, 2])})
        assert model.calls == 3
        model.train()
        model({'ids': torch.tensor([1, 2])})
        assert model.calls == 4 and not cache.data


def test_glip_cache_restores_label_metadata_and_copies_results():
    class Demo:
        model = SimpleNamespace(training=False)
        confidence_threshold = .6
        calls = 0
        def inference(self, image, caption):
            self.calls += 1
            self.entities = list(caption)
            self.plus = 1
            self.positive_map_label_to_token = {1: [0]}
            return {'labels': torch.tensor([1])}
    demo = Demo()
    cache = cache_glip(demo)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    demo.inference(image, ['chair'])['labels'].zero_()
    demo.inference(image, ['bed'])
    result = demo.inference(image, ['chair'])
    assert result['labels'].item() == 1 and demo.entities == ['chair']
    assert demo.calls == 2 and cache.hits == 1
    demo.inference(image + 1, ['chair'])
    assert demo.calls == 3


def test_sam_cache_restores_embedding_and_geometry():
    class Predictor:
        model = SimpleNamespace(training=False)
        calls = 0
        def set_image(self, image, image_format='RGB'):
            self.calls += 1
            self.features = torch.tensor([float(image.sum())])
            self.original_size = image.shape[:2]
            self.input_size = image.shape[:2]
            self.is_image_set = True
    predictor = Predictor()
    cache = cache_sam_images(predictor)
    image = np.ones((2, 3, 3), dtype=np.uint8)
    predictor.set_image(image)
    predictor.features.zero_()
    predictor.set_image(np.zeros((3, 4, 3), dtype=np.uint8))
    predictor.set_image(image)
    assert predictor.features.item() == 18 and predictor.original_size == (2, 3)
    assert predictor.calls == 2 and cache.hits == 1
