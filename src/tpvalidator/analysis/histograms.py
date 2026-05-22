from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import boost_histogram as bh
import hist
import numpy as np
import pandas as pd

type Hist1D = hist.Hist | bh.Histogram
type Quantiles = float | Sequence[float]




# def hist_mean_cov_uhi(
#     h, *, flow: bool = False
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Compute mean vector and covariance matrix for an N-D UHI histogram
#     using bin centers and bin contents as weights.

#     Parameters
#     ----------
#     h
#         Any UHI histogram (boost-histogram, hist, etc.)
#     flow
#         If False (recommended): ignore under/overflow bins entirely.
#         If True: include under/overflow bins ONLY in the total normalization;
#                  moments are still computed using the regular-bin region only
#                  (since flow bins do not have well-defined coordinates).

#     Returns
#     -------
#     mean : np.ndarray
#         Shape (ndim,) mean vector.
#     cov : np.ndarray
#         Shape (ndim, ndim) covariance matrix.
#     std : np.ndarray
#         Shape (ndim,) standard deviation per axis (sqrt(diag(cov))).
#     """
#     ndim = len(h.axes)
#     if ndim == 0:
#         raise ValueError("Histogram has no axes.")

#     w_core = np.asarray(h.values(flow=False), dtype=float)
#     core_sum = float(w_core.sum())

#     if flow:
#         w_total = float(np.asarray(h.values(flow=True), dtype=float).sum())
#     else:
#         w_total = core_sum

#     if w_total <= 0.0 or core_sum <= 0.0:
#         mean = np.full((ndim,), np.nan)
#         cov = np.full((ndim, ndim), np.nan)
#         std = np.full((ndim,), np.nan)
#         return mean, cov, std

#     centers = [np.asarray(ax.centers, dtype=float) for ax in h.axes]
#     shape = w_core.shape

#     # Broadcast each axis's centers to the full grid shape without meshgrid
#     bc = []
#     for i in range(ndim):
#         rshape = [1] * ndim
#         rshape[i] = shape[i]
#         bc.append(centers[i].reshape(rshape))

#     mean = np.empty((ndim,), dtype=float)
#     for i in range(ndim):
#         mean[i] = float(np.sum(w_core * bc[i]) / w_total)

#     exx = np.empty((ndim, ndim), dtype=float)
#     for i in range(ndim):
#         for j in range(i, ndim):
#             exx_ij = float(np.sum(w_core * bc[i] * bc[j]) / w_total)
#             exx[i, j] = exx_ij
#             exx[j, i] = exx_ij

#     cov = exx - np.outer(mean, mean)
#     std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
#     return mean, cov, std

# TODO: review 
def hist_mean_std_uhi(h, *, flow: bool = False) -> tuple[float, float]:
    """
    Compute mean and standard deviation of a 1D UHI-compatible histogram.

    Parameters
    ----------
    h
        Any UHI histogram (boost-histogram, hist, etc.)
    flow
        Whether to include under/overflow bins in the normalization.

    Returns
    -------
    mean, std
        Weighted mean and standard deviation.
    """
    axis = h.axes[0]
    centers = axis.centers
    values = np.asarray(h.values(flow=flow), dtype=float)
    if flow:
        values = values[1:-1]
    total = values.sum()
    if total <= 0:
        return np.nan, np.nan
    mean = np.sum(values * centers) / total
    var = np.sum(values * (centers - mean) ** 2) / total
    return mean, np.sqrt(var)

# TODO: review 
def hist_mean_std_uproot(h) -> dict[int, tuple[float, float]]:
    """
    Compute mean and std deviation for an uproot histogram (TH1, TH2).

    Returns
    -------
    dict mapping axis index → (mean, std).
    """
    arr = h.to_numpy()

    if len(arr) == 2:
        values, edges = arr
        centers = 0.5 * (edges[1:] + edges[:-1])
        total = values.sum()
        if total == 0:
            raise ValueError("Histogram is empty")
        mean = np.average(centers, weights=values)
        var = np.average((centers - mean) ** 2, weights=values)
        return {0: (mean, np.sqrt(var))}

    if len(arr) == 3:
        values, xedges, yedges = arr
        total = values.sum()
        if total == 0:
            raise ValueError("Histogram is empty")
        xcenters = 0.5 * (xedges[1:] + xedges[:-1])
        ycenters = 0.5 * (yedges[1:] + yedges[:-1])
        X, Y = np.meshgrid(xcenters, ycenters, indexing="ij")
        mean_x = (values * X).sum() / total
        mean_y = (values * Y).sum() / total
        var_x = (values * (X - mean_x) ** 2).sum() / total
        var_y = (values * (Y - mean_y) ** 2).sum() / total
        return {0: (mean_x, np.sqrt(var_x)), 1: (mean_y, np.sqrt(var_y))}

    raise NotImplementedError("Only 1D and 2D histograms are supported")


#------------------------------------------------------------------------------
# Histogram stats
#
def _project(h: hist.Hist, axis_index: int) -> np.ndarray:
    """Marginalise over all axes except `axis_index`."""
    sum_axes = tuple(i for i in range(h.ndim) if i != axis_index)
    return h.values().sum(axis=sum_axes)


def _axis_moments(counts: np.ndarray, centers: np.ndarray) -> tuple[float, float]:
    """
    Compute mean and standard deviation along a single axis
    from counts and bin centres. Returns (nan, nan) if histogram is empty.
    """
    if centers is None:
        raise TypeError("Mean and std are not defined for categorical axes.")

    total = counts.sum()
    if total == 0:
        return float("nan"), float("nan")

    mean    = np.dot(counts, centers)      / total
    mean_x2 = np.dot(counts, centers ** 2) / total
    std     = float(np.sqrt(max(mean_x2 - mean ** 2, 0.0)))

    return float(mean), std



def hist_stats(h: hist.Hist) -> list[tuple[float, float]]:
    """
    Compute mean and std per axis for a hist.Hist of any dimensionality.

    Returns
    -------
    List of (mean, std) tuples, one per axis in axis order.

    Example
    -------
    >>> hist_stats(mgr["pt_vs_eta"])
    [(42.3, 18.7), (0.1, 1.2)]
    """
    return [
        _axis_moments(_project(h, i), h.axes[i].centers)
        for i in range(h.ndim)
    ]

#
# Histogram stats
#------------------------------------------------------------------------------
def _auto_range_full(values: np.ndarray) -> tuple[float, float]:
    """
    Compute histogram range covering all finite values,
    rounded outward to 2 significant figures for clean labels.
    """
    values = values[np.isfinite(values)]
    lo     = float(values.min())
    hi     = float(values.max())

    def _round_out(x: float, direction: int) -> float:
        """Round to the nearest power of 10

                """
        if x == 0:
            return 0.0
        mag = 10 ** (np.floor(np.log10(abs(x))) - 1)
        return direction * np.ceil(abs(x) / mag) * mag

    lo = _round_out(lo, -1) if lo < 0 else -_round_out(-lo, -1)
    hi = _round_out(hi,  1)
    return lo, hi


def compute_regaxis_specs(series: pd.Series, bin_size: int, direction: Literal['left', 'right']='right'):
    """
    Compute the inclusive histogram binning range for a choice of bin size

    Args:
        series (pd.Series): _description_
        bin_size (int, optional): _description_. Defaults to 10.
        direction (Literal[&#39;left&#39;, &#39;right&#39;], optional): _description_. Defaults to 'right'.

    Returns:
        _type_: _description_
    """
    import math
    lo     = float(series.min())
    hi     = float(series.max())

    num_bins = math.floor((hi-lo)/bin_size)+1

    match direction:
        case 'right':
            hi = lo+bin_size*num_bins
        case 'left':
            lo = hi-bin_size*num_bins

    return num_bins, lo, hi


def linspace(series: pd.Series, step: int = 10) -> np.ndarray:
    """
    Return evenly-spaced bin edges spanning [series.min, series.max) with step `step`.
    """
    # lo, hi = _auto_range_full(series)

    lo     = float(series.min())
    hi     = float(series.max())

    return np.arange(lo, hi+step, step)





# TODO: Review and integrate
# def hist_and_sum(
#     df: pd.DataFrame,
#     cols: Sequence[str],
#     bins: int | Sequence[float],
# ) -> tuple[dict[str, hist.Hist], hist.Hist]:
#     """
#     Generate histograms for selected DataFrame columns and their sum.

#     Parameters
#     ----------
#     df
#         Input DataFrame.
#     cols
#         Column names to histogram.
#     bins
#         Number of bins or explicit bin edges.

#     Returns
#     -------
#     histograms
#         Dictionary mapping column name → Hist.
#     sum_hist
#         Histogram equal to the sum of all column histograms.
#     """
#     if isinstance(bins, int):
#         agg = df[list(cols)].agg(["min", "max"])
#         xmin = float(agg.loc["min"].min())
#         xmax = float(agg.loc["max"].max())
#         axis = hist.axis.Regular(bins, xmin, xmax, name="x")
#     else:
#         axis = hist.axis.Variable(bins, name="x")

#     histograms = {}
#     for c in cols:
#         h = hist.Hist(axis, storage=hist.Double())
#         h.fill(df[c].values)
#         histograms[c] = h

#     sum_hist = sum(histograms.values())
#     return histograms, sum_hist


def _extract_flow(
    h: Hist1D,
    *,
    flow: bool,
) -> tuple[float, np.ndarray, float, float, np.ndarray, float]:
    """
    Split a 1D histogram into (under, vals, over, under_var, vars_, over_var).

    When flow=False, under/over and their variances are 0.0.
    vars_ is zeros when the histogram has no variance storage.
    """
    vals_all = np.asarray(h.values(flow=flow), dtype=float)
    variances = h.variances(flow=flow) if flow else h.variances()

    if flow:
        under, vals, over = float(vals_all[0]), vals_all[1:-1], float(vals_all[-1])
        if variances is not None:
            vars_all = np.asarray(variances, dtype=float)
            under_var, vars_, over_var = float(vars_all[0]), vars_all[1:-1], float(vars_all[-1])
        else:
            under_var, vars_, over_var = 0.0, np.zeros_like(vals), 0.0
    else:
        under, over, under_var, over_var = 0.0, 0.0, 0.0, 0.0
        vals = vals_all
        vars_ = np.asarray(variances, dtype=float) if variances is not None else np.zeros_like(vals)

    return under, vals, over, under_var, vars_, over_var


def _build_cdf(
    h: Hist1D,
    *,
    flow: bool = False,
    use_density: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """
    Build a normalized CDF from a 1D histogram.

    Returns (cdf, edges, offset, tail), where:
      cdf    : normalized cumulative masses, shape (nbins,)
      edges  : bin edges, shape (nbins + 1,)
      offset : fraction of total weight in the underflow bin
      tail   : fraction of total weight in the overflow bin

    Returns None if the total weight is <= 0.
    """
    edges = h.axes[0].edges
    under, vals, over, *_ = _extract_flow(h, flow=flow)

    masses = vals * np.diff(edges) if use_density else vals
    total = under + over + float(masses.sum())

    if total <= 0.0:
        return None

    cdf = np.cumsum(masses) / total
    return cdf, edges, under / total, over / total


def quantiles(
    h: Hist1D,
    qs: Quantiles,
    *,
    flow: bool = False,
    use_density: bool = False,
) -> np.ndarray:
    """
    Compute quantiles for a 1D Scikit-HEP hist / boost-histogram histogram.

    Quantiles are linearly interpolated inside each bin.
    If flow=True, under/overflow bins contribute only to normalization.
    Returns NaNs if the total weight is <= 0.
    """
    qs_arr = np.atleast_1d(qs).astype(float)
    if np.any((qs_arr < 0.0) | (qs_arr > 1.0)):
        raise ValueError("Quantiles must be in [0, 1].")

    result = _build_cdf(h, flow=flow, use_density=use_density)
    if result is None:
        return np.full_like(qs_arr, np.nan)

    cdf, edges, offset, tail = result
    targets = offset + qs_arr * (1.0 - offset - tail)

    idx = np.searchsorted(cdf, targets, side="left")
    idx = np.clip(idx, 0, len(cdf) - 1)

    cdf_lo = np.where(idx > 0, cdf[idx - 1], 0.0)
    cdf_hi = cdf[idx]
    x_lo = edges[idx]
    x_hi = edges[idx + 1]

    t = np.where(cdf_hi > cdf_lo, (targets - cdf_lo) / (cdf_hi - cdf_lo), 0.0)
    return x_lo + t * (x_hi - x_lo)


def cumsum_hist(
    h: hist.Hist,
    direction: Literal["left", "right"] = "left",
    normalise: bool = False,
    flow: bool = False,
) -> hist.Hist:
    """
    Compute the cumulative sum of a 1D hist.Hist.

    Parameters
    ----------
    h         : source histogram
    direction : 'left'  → cumsum low-to-high (events below threshold)
                'right' → cumsum high-to-low (events above threshold)
    normalise : if True, scale to [0, 1]
    flow      : if True, include under/overflow bins in the cumulative sum.
                Underflow is added as a constant offset for 'left';
                overflow for 'right'. Both contribute to the normalisation total.

    Returns
    -------
    A new hist.Hist with the same axis as the source.
    """
    if h.ndim != 1:
        raise ValueError(f"Expected a 1D histogram, got {h.ndim}D.")

    under, vals, over, under_var, vars_, over_var = _extract_flow(h, flow=flow)

    if direction == "left":
        cum = np.cumsum(vals) + under
        cum_vars = np.cumsum(vars_) + under_var
        norm = cum[-1] + over
    elif direction == "right":
        cum = np.cumsum(vals[::-1])[::-1] + over
        cum_vars = np.cumsum(vars_[::-1])[::-1] + over_var
        norm = cum[0] + under
    else:
        raise ValueError(f"direction must be 'left' or 'right', got '{direction}'.")

    if normalise and norm != 0:
        cum_vars = cum_vars / (norm ** 2)
        cum = cum / norm

    axis = h.axes[0]
    out_axis = hist.axis.Variable(
        axis.edges,
        name=axis.name,
        label=getattr(axis, "label", None),
        underflow=False,
        overflow=False,
    )
    cum_hist = hist.Hist(out_axis, storage=hist.storage.Weight())
    cum_hist.view().value    = cum
    cum_hist.view().variance = cum_vars
    return cum_hist
