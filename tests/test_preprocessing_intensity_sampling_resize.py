import numpy as np
import pytest

from orthovision.preprocessing.config import PreprocessConfig
from orthovision.preprocessing.intensity import NormalizationError, prepare_slice
from orthovision.preprocessing.resize import letterbox_resize, stack_triplet_channels
from orthovision.preprocessing.sampling import (
    SamplingError,
    build_triplet,
    plan_series_samples,
    sample_locations,
)


def cfg_percentile(**kw):
    return PreprocessConfig(normalization_method="percentile", **kw)


def test_percentile_normalization_bounds_and_clipping():
    rng = np.random.default_rng(7)
    arr = rng.normal(loc=500.0, scale=100.0, size=(64, 64))
    arr[0, 0] = 1e6
    out, stats = prepare_slice(arr, photometric="MONOCHROME2", cfg=cfg_percentile())
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert stats.clip_low < stats.clip_high


def test_zscore_zero_mean_unit_std():
    rng = np.random.default_rng(3)
    arr = rng.normal(loc=200.0, scale=50.0, size=(128, 128))
    out, stats = prepare_slice(arr, photometric=None, cfg=PreprocessConfig(normalization_method="zscore"))
    assert abs(float(out.mean())) < 0.05
    assert abs(float(out.std()) - 1.0) < 0.1
    assert stats.std > 0


def test_constant_image_returns_zeros_not_nan():
    arr = np.full((32, 32), 77.0)
    for method in ("percentile", "zscore"):
        out, _ = prepare_slice(arr, photometric=None, cfg=PreprocessConfig(normalization_method=method))
        assert not np.isnan(out).any()
        assert float(out.max()) == 0.0


def test_monochrome1_inverted():
    ramp = np.linspace(0, 1000, 64 * 64).reshape(64, 64)
    normal_out, _ = prepare_slice(ramp, photometric="MONOCHROME2", cfg=cfg_percentile())
    inverted_out, stats = prepare_slice(ramp, photometric="MONOCHROME1", cfg=cfg_percentile())
    assert stats.inverted_monochrome1
    assert np.allclose(normal_out + inverted_out, 1.0, atol=1e-5)


def test_unknown_method_raises():
    with pytest.raises(NormalizationError):
        prepare_slice(np.ones((8, 8)), photometric=None, cfg=PreprocessConfig(normalization_method="histogram"))


def test_sample_locations_even_with_margin():
    cfg = PreprocessConfig(num_samples_per_plane=6, boundary_margin=0.06)
    locs = sample_locations(40, cfg)
    assert len(locs) == len(set(locs)) == 6
    n_minus_1 = 39
    assert locs[0] == round(0.06 * n_minus_1)
    assert locs[-1] == round(0.94 * n_minus_1)
    diffs = np.diff(locs)
    assert diffs.min() >= 1


def test_sample_locations_explicit_override_and_dedupe():
    cfg = PreprocessConfig(sample_positions=(0.0, 0.5, 0.5, 1.0), num_samples_per_plane=6)
    locs = sample_locations(11, cfg)
    assert locs == [0, 5, 10]


def test_triplet_channel_order_index_strategy():
    cfg = PreprocessConfig()
    plan = build_triplet(4, 20, cfg, [float(i) for i in range(20)])
    assert plan.slice_indices == (3, 4, 5)
    assert plan.physical_positions_mm == (3.0, 4.0, 5.0)


def test_triplet_skips_boundary_centers():
    plans = plan_series_samples(8, None, PreprocessConfig(num_samples_per_plane=4, boundary_margin=0.06))
    for plan in plans:
        p, c, n = plan.slice_indices
        assert 0 <= p < c < n <= 7


def test_physical_distance_strategy_picks_mm_neighbours():
    positions = [0.0, 3.0, 6.0, 9.0, 12.0]
    cfg = PreprocessConfig(neighbor_strategy="physical_distance", neighbor_target_spacing_mm=3.0)
    plan = build_triplet(2, len(positions), cfg, positions)
    assert plan.slice_indices == (1, 2, 3)
    assert plan.physical_positions_mm == (3.0, 6.0, 9.0)


def test_insufficient_slices_yield_no_triplets():
    plans = plan_series_samples(2, None, PreprocessConfig(num_samples_per_plane=6))
    assert plans == []


def test_invalid_offsets_rejected():
    with pytest.raises(SamplingError):
        build_triplet(1, 10, PreprocessConfig(neighbor_offsets=(1, 0, -1)), None)


def test_letterbox_preserves_aspect_and_records_meta():
    cfg = PreprocessConfig(image_size=(224, 224))
    img = np.zeros((100, 400), dtype=np.float32)
    out, meta = letterbox_resize(img, cfg)
    assert out.shape == (224, 224)
    assert meta.scale == pytest.approx(224 / 400)
    used_h = int(round(100 * meta.scale))
    assert meta.pad_top + used_h + meta.pad_bottom == 224
    assert meta.pad_top > 0 and meta.pad_bottom > 0
    assert meta.pad_left == 0 and meta.pad_right == 0


def test_non_square_upscale_letterbox():
    cfg = PreprocessConfig(image_size=(64, 64))
    img = np.ones((30, 20), dtype=np.float32)
    out, meta = letterbox_resize(img, cfg)
    assert out.shape == (64, 64)
    assert meta.scale == pytest.approx(64 / 30)
    used_w = int(round(20 * meta.scale))
    assert meta.pad_top == 0 and meta.pad_bottom == 0
    assert meta.pad_left + used_w + meta.pad_right == 64
    assert float(out[:, : meta.pad_left].max()) == 0.0


def test_stack_triplet_channels_shape():
    cfg = PreprocessConfig(image_size=(32, 32))
    channels = [np.random.default_rng(i).random((17, 29)).astype(np.float32) for i in range(3)]
    stacked, metas = stack_triplet_channels(channels, cfg)
    assert stacked.shape == (3, 32, 32)
    assert len(metas) == 3
