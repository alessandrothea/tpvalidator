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
        print(f"\nnum_bins={num_bins} (exp 5), lo={lo} (exp 0.0), hi={hi} (exp 25.0)")
        assert num_bins == 5
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(25.0)

    def test_direction_left(self):
        s = pd.Series([0, 10, 20])
        num_bins, lo, hi = compute_regaxis_specs(s, bin_size=10, direction="left")
        # num_bins = floor(20/10)+1 = 3; lo = hi - 10*3 = 20-30 = -10
        print(f"\nnum_bins={num_bins} (exp 3), lo={lo} (exp -10.0), hi={hi} (exp 20.0)")
        assert num_bins == 3
        assert lo == pytest.approx(-10.0)
        assert hi == pytest.approx(20.0)

    def test_bin_count_covers_range(self):
        s = pd.Series([1, 99])
        num_bins, lo, hi = compute_regaxis_specs(s, bin_size=10)
        print(f"\nhi - lo={hi - lo}, num_bins * 10={num_bins * 10.0}")
        assert (hi - lo) == pytest.approx(num_bins * 10.0)


class TestLinspace:
    def test_step_respected(self):
        s = pd.Series([0.0, 10.0])
        edges = linspace(s, step=2)
        diffs = np.diff(edges)
        print(f"\nedge diffs: {diffs}, expected all 2.0")
        assert np.all(diffs == pytest.approx(2.0))

    def test_covers_range(self):
        s = pd.Series([3.0, 13.0])
        edges = linspace(s, step=5)
        print(f"\nedges[0]={edges[0]} (exp 3.0), edges[-1]={edges[-1]} (expected >= 13.0)")
        assert edges[0] == pytest.approx(3.0)
        assert edges[-1] >= 13.0


class TestCumsumHist:
    def test_left_monotone(self, uniform_hist_1d):
        c = cumsum_hist(uniform_hist_1d, direction="left")
        diffs = np.diff(c.values())
        print(f"\ndiffs (left cumsum, expect all >= 0): min={diffs.min():.4f}, max={diffs.max():.4f}")
        assert np.all(diffs >= 0)

    def test_right_monotone(self, uniform_hist_1d):
        c = cumsum_hist(uniform_hist_1d, direction="right")
        diffs = np.diff(c.values())
        print(f"\ndiffs (right cumsum, expect all <= 0): min={diffs.min():.4f}, max={diffs.max():.4f}")
        assert np.all(diffs <= 0)

    def test_preserves_axis_name(self, uniform_hist_1d):
        c = cumsum_hist(uniform_hist_1d)
        name = c.axes[0].name
        print(f"\naxis name: '{name}', expected: 'x'")
        assert name == "x"

    def test_rejects_nd(self, uniform_hist_2d):
        with pytest.raises(ValueError, match="1D"):
            cumsum_hist(uniform_hist_2d)

    def test_flow_left(self, uniform_hist_1d):
        c = cumsum_hist(uniform_hist_1d, direction="left")
        print(f"\nc.ndim: {c.ndim}, expected: 1")
        assert c.ndim == 1

    def test_weight_storage(self, uniform_hist_1d):
        c = cumsum_hist(uniform_hist_1d)
        fields = c.view().dtype.names
        print(f"\ndtype fields: {fields} (expect named fields for Weight storage)")
        assert fields is not None  # Weight storage has named fields

    # --- total stored in opposing flow bin ---

    def test_overflow_bin_stores_norm(self):
        h = hist.Hist(hist.axis.Regular(10, 0, 10, name="x"))
        h.fill(x=np.concatenate([np.linspace(0.5, 9.5, 90), np.full(10, 15.0)]))
        c = cumsum_hist(h, direction="left")
        expected = h.values(flow=True).sum()  # grand total including overflow
        got = c.view(flow=True).value[-1]
        print(f"\noverflow bin (norm, left): {got}")
        print(f"expected (grand total):    {expected}")
        assert got == pytest.approx(expected)

    def test_underflow_bin_stores_norm(self):
        h = hist.Hist(hist.axis.Regular(10, 0, 10, name="x"))
        h.fill(x=np.concatenate([np.linspace(0.5, 9.5, 90), np.full(10, -5.0)]))
        c = cumsum_hist(h, direction="right")
        expected = h.values(flow=True).sum()  # grand total including underflow
        got = c.view(flow=True).value[0]
        print(f"\nunderflow bin (norm, right): {got}")
        print(f"expected (grand total):      {expected}")
        assert got == pytest.approx(expected)


class TestCumsumHistNd:
    def test_preserves_ndim(self, uniform_hist_2d):
        c = cumsum_hist_nd(uniform_hist_2d, axis="x")
        print(f"\nc.ndim: {c.ndim}, expected: 2")
        assert c.ndim == 2

    def test_unknown_axis_name_raises(self, uniform_hist_2d):
        with pytest.raises(ValueError, match="nope"):
            cumsum_hist_nd(uniform_hist_2d, axis="nope")

    def test_out_of_range_index_raises(self, uniform_hist_2d):
        with pytest.raises(ValueError):
            cumsum_hist_nd(uniform_hist_2d, axis=5)

    def test_flow(self, uniform_hist_2d):
        c = cumsum_hist_nd(uniform_hist_2d, axis="x")
        print(f"\nc.ndim: {c.ndim}, expected: 2")
        assert c.ndim == 2

    # --- count-preservation tests ---

    def test_total_preserved_in_last_bin_left_no_overflow(self, uniform_hist_2d):
        c = cumsum_hist_nd(uniform_hist_2d, axis="x", direction="left")
        expected = uniform_hist_2d.values().sum(axis=0)
        print(f"\nlast bin of cumsum (axis=x, left): {c.values()[-1, :]}")
        print(f"expected (sum over x):              {expected}")
        assert np.allclose(c.values()[-1, :], expected)

    def test_total_preserved_in_first_bin_right_no_overflow(self, uniform_hist_2d):
        c = cumsum_hist_nd(uniform_hist_2d, axis="x", direction="right")
        expected = uniform_hist_2d.values().sum(axis=0)
        print(f"\nfirst bin of cumsum (axis=x, right): {c.values()[0, :]}")
        print(f"expected (sum over x):                {expected}")
        assert np.allclose(c.values()[0, :], expected)



class TestQuantiles:
    def _uniform_hist(self):
        h = hist.Hist(hist.axis.Regular(100, 0, 100, name="x"))
        h.fill(x=np.linspace(0.5, 99.5, 1000))
        return h

    def test_median(self):
        h = self._uniform_hist()
        q50 = quantiles(h, 0.5)
        print(f"\nq50: {q50[0]:.4f}, expected: ~50.0")
        assert q50[0] == pytest.approx(50.0, abs=1.0)

    def test_zero_quantile(self):
        h = self._uniform_hist()
        q0 = quantiles(h, 0.0)
        print(f"\nq0: {q0[0]:.4f}, expected: ~0.0")
        assert q0[0] == pytest.approx(0.0, abs=1.0)

    def test_one_quantile(self):
        h = self._uniform_hist()
        q1 = quantiles(h, 1.0)
        print(f"\nq1: {q1[0]:.4f}, expected: ~100.0")
        assert q1[0] == pytest.approx(100.0, abs=1.0)

    def test_empty_histogram_returns_nan(self):
        h = hist.Hist(hist.axis.Regular(10, 0, 10, name="x"))
        result = quantiles(h, 0.5)
        print(f"\nquantile(empty, 0.5): {result[0]}, expected: nan")
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
        mean, _ = stats[0]
        print(f"\nlen(stats)={len(stats)} (exp 1), mean={mean:.4f} (exp ~5.0)")
        assert len(stats) == 1
        assert mean == pytest.approx(5.0, abs=0.6)

    def test_2d_returns_two_entries(self, uniform_hist_2d):
        stats = hist_stats(uniform_hist_2d)
        print(f"\nlen(stats): {len(stats)}, expected: 2")
        assert len(stats) == 2

    def test_2d_means_in_range(self, uniform_hist_2d):
        stats = hist_stats(uniform_hist_2d)
        mean_x, _ = stats[0]
        mean_y, _ = stats[1]
        print(f"\nmean_x={mean_x:.4f} (exp 4-6), mean_y={mean_y:.4f} (exp 2-3)")
        assert 4.0 < mean_x < 6.0
        assert 2.0 < mean_y < 3.0
