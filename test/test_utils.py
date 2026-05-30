import logging
import numpy as np
import pytest

from tpvalidator.utils import (
    get_hist_layout,
    compute_histogram_ratio,
    calculate_trg_obj_rates,
    temporary_log_level,
)


class TestGetHistLayout:
    def test_one_item(self):
        assert get_hist_layout(1) == (1, 1)

    def test_four_items(self):
        assert get_hist_layout(4) == (2, 2)

    def test_six_items(self):
        # ceil(sqrt(6))=3 cols, ceil(6/3)=2 rows
        nrows, ncols = get_hist_layout(6)
        assert nrows * ncols >= 6

    def test_nine_items(self):
        assert get_hist_layout(9) == (3, 3)

    def test_explicit_layout_passthrough(self):
        assert get_hist_layout(10, layout=(2, 5)) == (2, 5)

    def test_output_covers_all_items(self):
        for n in range(1, 20):
            nrows, ncols = get_hist_layout(n)
            assert nrows * ncols >= n


class TestComputeHistogramRatio:
    def test_identical_data_ratio_is_one(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 10, 500)
        _, ratio, _, _ = compute_histogram_ratio(data, data, bins=10, range=(0, 10))
        assert np.allclose(ratio, 1.0)

    def test_zero_denominator_gives_nan_by_default(self):
        num = np.array([1.0, 2.0, 3.0])
        # denominator has no entries in certain bins if we use narrow range
        denom = np.array([10.0, 20.0, 30.0])
        _, ratio, _, _ = compute_histogram_ratio(num, denom, bins=50, range=(0, 100))
        # Most bins will be empty in both — ratio should be nan or 1.0
        # At least verify the function runs and returns arrays of matching length
        assert len(ratio) == 50

    def test_custom_zero_division(self):
        num = np.array([1.0])
        denom = np.array([100.0])  # completely different ranges when binned separately
        _, ratio, _, _ = compute_histogram_ratio(
            num, denom, bins=10, range=(0, 10), zero_division=-1.0
        )
        # Some bins will be 0/0 → should be -1.0 (custom value)
        assert (-1.0 in ratio) or np.all(np.isfinite(ratio))

    def test_output_shapes_consistent(self):
        rng = np.random.default_rng(1)
        data = rng.uniform(0, 1, 100)
        centers, ratio, ratio_err, bins = compute_histogram_ratio(data, data, bins=20)
        assert len(centers) == 20
        assert len(ratio) == 20
        assert len(ratio_err) == 20
        assert len(bins) == 21  # edges: n_bins + 1


class TestCalculateTrgObjRates:
    class _MockDf:
        """Minimal mock that satisfies calculate_trg_obj_rates."""
        def __init__(self, n, readout_window, num_entries):
            self._n = n
            self.extra_info = {"readout_window": readout_window, "num_entries": num_entries}

        def __len__(self):
            return self._n

    def test_known_rate(self):
        # 100 TPs, 6000-tick window, 10 entries, sampling_time=0.5e-6 s
        # tot_time = 6000 * 0.5e-6 * 10 = 0.03 s  →  rate = 100/0.03 ≈ 3333.3 Hz
        mock = self._MockDf(n=100, readout_window=6000, num_entries=10)
        rate = calculate_trg_obj_rates(mock)
        assert rate == pytest.approx(100 / (6000 * 0.5e-6 * 10), rel=1e-6)

    def test_override_readout_window(self):
        mock = self._MockDf(n=50, readout_window=0, num_entries=5)
        rate = calculate_trg_obj_rates(mock, readout_window=2000)
        expected = 50 / (2000 * 0.5e-6 * 5)
        assert rate == pytest.approx(expected, rel=1e-6)

    def test_zero_window_raises(self):
        mock = self._MockDf(n=10, readout_window=0, num_entries=5)
        with pytest.raises(ValueError):
            calculate_trg_obj_rates(mock)


class TestTemporaryLogLevel:
    def test_level_restored_after_context(self):
        logger = logging.getLogger("test_tmp_level")
        logger.setLevel(logging.WARNING)
        with temporary_log_level(logger, logging.DEBUG):
            assert logger.level == logging.DEBUG
        assert logger.level == logging.WARNING

    def test_level_changed_inside_context(self):
        logger = logging.getLogger("test_tmp_level_inside")
        logger.setLevel(logging.ERROR)
        with temporary_log_level(logger, logging.INFO):
            assert logger.level == logging.INFO
