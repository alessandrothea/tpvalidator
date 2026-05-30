import numpy as np
import pandas as pd
import pytest
import hist

from tpvalidator.analysis.histograms import (
    compute_regaxis_specs,
    linspace,
    cumsum_hist,
    cumsum_hist_nd,
    quantiles,
    hist_stats,
)


class TestComputeRegaxisSpecs:
    def test_even_range(self):
        s = pd.Series([0, 5, 10, 15, 20])
        num_bins, lo, hi = compute_regaxis_specs(s, bin_size=5)
        # floor((20-0)/5)+1 = 5; hi = 0 + 5*5 = 25
        assert num_bins == 5
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(25.0)

    def test_direction_left(self):
        s = pd.Series([0, 10, 20])
        num_bins, lo, hi = compute_regaxis_specs(s, bin_size=10, direction="left")
        # num_bins = floor(20/10)+1 = 3; lo = hi - 10*3 = 20-30 = -10
        assert num_bins == 3
        assert lo == pytest.approx(-10.0)
        assert hi == pytest.approx(20.0)

    def test_bin_count_covers_range(self):
        s = pd.Series([1, 99])
        num_bins, lo, hi = compute_regaxis_specs(s, bin_size=10)
        assert (hi - lo) == pytest.approx(num_bins * 10.0)


class TestLinspace:
    def test_step_respected(self):
        s = pd.Series([0.0, 10.0])
        edges = linspace(s, step=2)
        assert np.all(np.diff(edges) == pytest.approx(2.0))

    def test_covers_range(self):
        s = pd.Series([3.0, 13.0])
        edges = linspace(s, step=5)
        assert edges[0] == pytest.approx(3.0)
        assert edges[-1] >= 13.0


class TestCumsumHist:
    def test_left_monotone(self, uniform_hist_1d):
        c = cumsum_hist(uniform_hist_1d, direction="left")
        assert np.all(np.diff(c.values()) >= 0)

    def test_right_monotone(self, uniform_hist_1d):
        c = cumsum_hist(uniform_hist_1d, direction="right")
        assert np.all(np.diff(c.values()) <= 0)

    def test_left_normalised_ends_at_one(self, uniform_hist_1d):
        c = cumsum_hist(uniform_hist_1d, direction="left", normalise=True)
        assert c.values()[-1] == pytest.approx(1.0)

    def test_right_normalised_starts_at_one(self, uniform_hist_1d):
        c = cumsum_hist(uniform_hist_1d, direction="right", normalise=True)
        assert c.values()[0] == pytest.approx(1.0)

    def test_preserves_axis_name(self, uniform_hist_1d):
        c = cumsum_hist(uniform_hist_1d)
        assert c.axes[0].name == "x"

    def test_rejects_nd(self, uniform_hist_2d):
        with pytest.raises(ValueError, match="1D"):
            cumsum_hist(uniform_hist_2d)

    def test_flow_left(self, uniform_hist_1d):
        c = cumsum_hist(uniform_hist_1d, direction="left", flow=True)
        assert c.ndim == 1

    def test_weight_storage(self, uniform_hist_1d):
        c = cumsum_hist(uniform_hist_1d)
        assert c.view().dtype.names is not None  # Weight storage has named fields


class TestCumsumHistNd:
    def test_left_normalised_last_slice_is_one(self, uniform_hist_2d):
        c = cumsum_hist_nd(uniform_hist_2d, axis="x", direction="left", normalise=True)
        assert np.allclose(c.values()[-1, :], 1.0, atol=1e-10)

    def test_right_normalised_first_slice_is_one(self, uniform_hist_2d):
        c = cumsum_hist_nd(uniform_hist_2d, axis="x", direction="right", normalise=True)
        assert np.allclose(c.values()[0, :], 1.0, atol=1e-10)

    def test_axis_by_index(self, uniform_hist_2d):
        c = cumsum_hist_nd(uniform_hist_2d, axis=0, direction="left", normalise=True)
        assert np.allclose(c.values()[-1, :], 1.0, atol=1e-10)

    def test_second_axis(self, uniform_hist_2d):
        c = cumsum_hist_nd(uniform_hist_2d, axis="y", direction="left", normalise=True)
        assert np.allclose(c.values()[:, -1], 1.0, atol=1e-10)

    def test_preserves_ndim(self, uniform_hist_2d):
        c = cumsum_hist_nd(uniform_hist_2d, axis="x")
        assert c.ndim == 2

    def test_matches_1d_on_projected_histogram(self, uniform_hist_2d):
        h1 = uniform_hist_2d[sum, :]  # project to y-axis
        cs1 = cumsum_hist(h1, direction="left", normalise=True)
        cs_nd = cumsum_hist_nd(h1, axis="y", direction="left", normalise=True)
        assert np.allclose(cs1.values(), cs_nd.values())

    def test_unknown_axis_name_raises(self, uniform_hist_2d):
        with pytest.raises(ValueError, match="nope"):
            cumsum_hist_nd(uniform_hist_2d, axis="nope")

    def test_out_of_range_index_raises(self, uniform_hist_2d):
        with pytest.raises(ValueError):
            cumsum_hist_nd(uniform_hist_2d, axis=5)

    def test_flow(self, uniform_hist_2d):
        c = cumsum_hist_nd(uniform_hist_2d, axis="x", flow=True)
        assert c.ndim == 2


class TestQuantiles:
    def _uniform_hist(self):
        h = hist.Hist(hist.axis.Regular(100, 0, 100, name="x"))
        h.fill(x=np.linspace(0.5, 99.5, 1000))
        return h

    def test_median(self):
        h = self._uniform_hist()
        q50 = quantiles(h, 0.5)
        assert q50[0] == pytest.approx(50.0, abs=1.0)

    def test_zero_quantile(self):
        h = self._uniform_hist()
        q0 = quantiles(h, 0.0)
        assert q0[0] == pytest.approx(0.0, abs=1.0)

    def test_one_quantile(self):
        h = self._uniform_hist()
        q1 = quantiles(h, 1.0)
        assert q1[0] == pytest.approx(100.0, abs=1.0)

    def test_empty_histogram_returns_nan(self):
        h = hist.Hist(hist.axis.Regular(10, 0, 10, name="x"))
        result = quantiles(h, 0.5)
        assert np.isnan(result[0])

    def test_invalid_quantile_raises(self):
        h = self._uniform_hist()
        with pytest.raises(ValueError):
            quantiles(h, 1.5)


class TestHistStats:
    def test_1d_known_mean(self):
        h = hist.Hist(hist.axis.Regular(10, 0, 10, name="x"))
        h.fill(x=np.full(100, 5.0))
        stats = hist_stats(h)
        assert len(stats) == 1
        mean, _ = stats[0]
        assert mean == pytest.approx(5.0, abs=0.6)

    def test_2d_returns_two_entries(self, uniform_hist_2d):
        stats = hist_stats(uniform_hist_2d)
        assert len(stats) == 2

    def test_2d_means_in_range(self, uniform_hist_2d):
        stats = hist_stats(uniform_hist_2d)
        mean_x, _ = stats[0]
        mean_y, _ = stats[1]
        assert 4.0 < mean_x < 6.0
        assert 2.0 < mean_y < 3.0
