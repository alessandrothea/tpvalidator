from typing import Tuple, Optional, Union, Sequence, List, Dict
import numpy as np
import pandas as pd

from collections.abc import Sequence
import hist
import boost_histogram as bh


type Hist1D = hist.Hist | bh.Histogram
type Quantiles = float | Sequence[float]


def hist_compute_ratio(
    numerator_data: Union[np.ndarray, Sequence[float]],
    denominator_data: Union[np.ndarray, Sequence[float]],
    bins: Union[int, Sequence[float]] = 50,
    range: Optional[Tuple[float, float]] = None,
    zero_division: float = np.nan
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the bin-wise ratio of two histograms from raw data arrays,
    including propagated Poisson errors.

    Parameters:
    ----------
    numerator_data : array-like
        Raw data for the numerator histogram.
    denominator_data : array-like
        Raw data for the denominator histogram.
    bins : int or sequence of scalars
        Number of bins or bin edges to use (passed to np.histogram).
    range : tuple, optional
        Lower and upper range of the bins.
    zero_division : float
        Value to use where division by zero occurs.

    Returns:
    -------
    bin_centers : np.ndarray
        Centers of the bins.
    ratio : np.ndarray
        Ratio of counts (numerator / denominator) per bin.
    ratio_err : np.ndarray
        Propagated error on the ratio per bin.
    bins : np.ndarray
        Bin edges used.
    """
    num_counts, bins = np.histogram(numerator_data, bins=bins, range=range)
    denom_counts, _ = np.histogram(denominator_data, bins=bins, range=range)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.true_divide(num_counts, denom_counts)
        ratio[~np.isfinite(ratio)] = zero_division

        # Assume Poisson: σ = sqrt(N)
        num_err = np.sqrt(num_counts)
        denom_err = np.sqrt(denom_counts)

        # Avoid divide-by-zero in error propagation
        safe_num = np.maximum(num_counts, 1)
        safe_denom = np.maximum(denom_counts, 1)

        ratio_err = ratio * np.sqrt(
            (num_err / safe_num)**2 + (denom_err / safe_denom)**2
        )
        ratio_err[~np.isfinite(ratio_err)] = 0

    return bin_centers, ratio, ratio_err, bins


def hist_mean_std_uhi(h, *, flow: bool = False):
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

    # Bin centers (UHI attribute)
    centers = axis.centers

    # Bin contents (sum of weights or counts)
    values = np.asarray(h.values(flow=flow), dtype=float)

    if flow:
        # Drop under/overflow for x-based quantities
        values = values[1:-1]

    total = values.sum()
    if total <= 0:
        return np.nan, np.nan

    mean = np.sum(values * centers) / total
    var = np.sum(values * (centers - mean) ** 2) / total

    return mean, np.sqrt(var)


def hist_mean_cov_uhi(h, *, flow: bool = False):
    """
    Compute mean vector and covariance matrix for an N-D UHI histogram
    using bin centers and bin contents as weights.

    Parameters
    ----------
    h
        Any UHI histogram (boost-histogram, hist, etc.)
    flow
        If False (recommended): ignore under/overflow bins entirely.
        If True: include under/overflow bins ONLY in the total normalization;
                 moments are still computed using the regular-bin region only
                 (since flow bins do not have well-defined coordinates).

    Returns
    -------
    mean : np.ndarray
        Shape (ndim,) mean vector.
    cov : np.ndarray
        Shape (ndim, ndim) covariance matrix.
    std : np.ndarray
        Shape (ndim,) standard deviation per axis (sqrt(diag(cov))).
    """
    ndim = len(h.axes)
    if ndim == 0:
        raise ValueError("Histogram has no axes.")
    if ndim == 1:
        # Still works, but keep behaviour consistent
        pass

    # Regular-bin contents (no flow bins) are where coordinates are defined
    w_core = np.asarray(h.values(flow=False), dtype=float)
    core_sum = float(w_core.sum())

    if flow:
        # Total includes flow bins, but moments use only core bins
        w_total = float(np.asarray(h.values(flow=True), dtype=float).sum())
    else:
        w_total = core_sum

    if w_total <= 0.0 or core_sum <= 0.0:
        mean = np.full((ndim,), np.nan)
        cov = np.full((ndim, ndim), np.nan)
        std = np.full((ndim,), np.nan)
        return mean, cov, std

    # Bin centers for each axis
    centers = [np.asarray(ax.centers, dtype=float) for ax in h.axes]
    shape = w_core.shape  # (n0, n1, ..., n_{ndim-1})

    # Helper: broadcast centers[i] to full grid shape without making a meshgrid
    def broadcast_centers(i: int) -> np.ndarray:
        rshape = [1] * ndim
        rshape[i] = shape[i]
        return centers[i].reshape(rshape)

    denom = w_total  # normalization (optionally includes flow)

    # Mean vector
    mean = np.empty((ndim,), dtype=float)
    for i in range(ndim):
        xi = broadcast_centers(i)
        mean[i] = float(np.sum(w_core * xi) / denom)

    # Second moments E[x_i x_j]
    exx = np.empty((ndim, ndim), dtype=float)
    for i in range(ndim):
        xi = broadcast_centers(i)
        for j in range(i, ndim):
            xj = broadcast_centers(j)
            exx_ij = float(np.sum(w_core * xi * xj) / denom)
            exx[i, j] = exx_ij
            exx[j, i] = exx_ij

    # Covariance: cov = E[xx^T] - mu mu^T
    cov = exx - np.outer(mean, mean)
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    return mean, cov, std



def hist_mean_std(h) -> Dict:
    """
    Compute statistical mean and std deviation for an uproot histogram object
    (TH1, TH2, ...). Returns a dict {axis_index: (mean, stddev)}.
    """
    arr = h.to_numpy()

    # 1D histogram
    if len(arr) == 2:
        values, edges = arr
        centers = 0.5 * (edges[1:] + edges[:-1])
        total = values.sum()
        if total == 0:
            raise ValueError("Histogram is empty")

        mean = np.average(centers, weights=values)
        var = np.average((centers - mean)**2, weights=values)
        return {0: (mean, np.sqrt(var))}

    # 2D histogram
    elif len(arr) == 3:
        values, xedges, yedges = arr
        total = values.sum()
        if total == 0:
            raise ValueError("Histogram is empty")

        xcenters = 0.5 * (xedges[1:] + xedges[:-1])
        ycenters = 0.5 * (yedges[1:] + yedges[:-1])

        X, Y = np.meshgrid(xcenters, ycenters, indexing="ij")

        mean_x = (values * X).sum() / total
        mean_y = (values * Y).sum() / total

        var_x = (values * (X - mean_x)**2).sum() / total
        var_y = (values * (Y - mean_y)**2).sum() / total

        return {
            0: (mean_x, np.sqrt(var_x)),
            1: (mean_y, np.sqrt(var_y)),
        }

    else:
        raise NotImplementedError("Only 1D and 2D histograms are supported for now")
    

def calculate_natural_bins( series : pd.Series, downsampling=10 ) -> List[float]:
    """Calculates a sort of "natural" binning of a pd.Series of discrete values,
    i.e. evenly spaced bins between the series min and max

    Args:
        series (pd.Series): _description_
        downsampling (int, optional): _description_. Defaults to 10.

    Returns:
        List(float): List of edges
    """


    x_min=series.min()
    x_max=series.max()

    # x_range=(x_max-x_min)
    # n_bins=int(x_range)//downsampling
    # dx = x_range/n_bins

    # bins = [ (x_min + i*dx) for i in range(n_bins+1)]

    # return bins

    return np.arange(x_min, x_max, downsampling)




def hist_and_sum_legacy(df: pd.DataFrame, cols: List[int], bins) -> Tuple[np.histogram, Dict[int, np.histogram]]:
    """Utility function that generates the histogram selected columns and the sums of all of them

    TODO: This code is 

    Args:
        df (pd.Dataframe): input Dataframe
        cols (List): list of columns to generate histograms from
        bins (_type_): _description_

    Returns:
        Tuple[np.histogram, Dict[int, np.histogram]]: _description_
    """
    histograms = {}
    for c in cols:
        hist, _ = np.histogram(df[c], bins=bins)
        histograms[c] = hist

    sum_hist = np.sum(list(histograms.values()), axis=0)
    return histograms, sum_hist




def hist_and_sum(
    df: pd.DataFrame,
    cols: Sequence[str],
    bins: int | Sequence[float],
) -> Tuple[Dict[str, hist.Hist], hist.Hist]:
    """
    Generate histograms for selected DataFrame columns and their sum
    using Scikit-HEP histograms.

    Parameters
    ----------
    df
        Input DataFrame.
    cols
        Column names to histogram.
    bins
        Number of bins or explicit bin edges.

    Returns
    -------
    histograms
        Dictionary mapping column name → Hist.
    sum_hist
        Histogram equal to the sum of all column histograms.
    """

    # --- Define a common axis -----------------------------------------------
    if isinstance(bins, int):
        xmin = df[cols].min().min()
        xmax = df[cols].max().max()
        axis = hist.Regular(bins, xmin, xmax, name="x")
    else:
        axis =hist.Regular.from_edges(bins, name="x")

    # --- Build one histogram per column -------------------------------------
    histograms = {}

    for c in cols:
        h = hist.Hist(axis, storage=hist.Double())
        h.fill(df[c].values)
        histograms[c] = h

    # --- Sum histograms (axis compatibility is checked automatically) -------
    sum_hist = sum(histograms.values())

    return histograms, sum_hist




def hist_quantiles(
    h: Hist1D,
    qs: Quantiles,
    *,
    flow: bool = False,
    use_density: bool = False,
) -> np.ndarray:
    """
    Compute quantiles for a 1D Scikit-HEP hist / boost-histogram histogram.

    The algorithm:
      1. Interpret bin contents as a mass distribution (counts or weights)
      2. Build a cumulative distribution function (CDF)
      3. Invert the CDF to obtain x-values for the requested quantiles

    Notes:
      - Quantiles are linearly interpolated inside each bin
      - If flow=True, under/overflow bins contribute only to normalization
      - Returns NaNs if the total weight is <= 0
    """

    # --- Normalize and validate requested quantiles -------------------------
    # Ensure we always work with an array of quantiles and check bounds
    qs_arr = np.atleast_1d(qs).astype(float)

    if np.any((qs_arr < 0.0) | (qs_arr > 1.0)):
        raise ValueError("Quantiles must be in [0, 1].")

    # --- Extract bin geometry ------------------------------------------------
    # Use the single axis of the 1D histogram
    # edges has shape (nbins + 1,)
    axis = h.axes[0]
    edges = axis.edges

    # --- Extract bin contents ------------------------------------------------
    # values(flow=True) includes [underflow, bins..., overflow]
    # values(flow=False) includes only regular bins
    vals_all = np.asarray(h.values(flow=flow), dtype=float)

    if flow:
        # Separate under/overflow from regular bins
        under = float(vals_all[0])
        over = float(vals_all[-1])
        vals = vals_all[1:-1]
    else:
        # Ignore under/overflow completely
        under = 0.0
        over = 0.0
        vals = vals_all

    # --- Convert bin contents to bin "masses" --------------------------------
    # If values represent densities, integrate them over the bin width
    # Otherwise treat values as total weight per bin
    widths = np.diff(edges)
    masses = vals * widths if use_density else vals

    # Total normalization (including flow bins if requested)
    total = under + over + float(masses.sum())

    if total <= 0.0:
        # No meaningful quantiles can be computed
        return np.full_like(qs_arr, np.nan)

    # --- Build the cumulative distribution function (CDF) -------------------
    # CDF[i] = fraction of total weight up to and including bin i
    cdf = np.cumsum(masses) / total

    # Fraction of probability sitting in under/overflow
    offset = under / total
    tail = over / total

    # Map requested quantiles into the regular-bin CDF range
    targets = offset + qs_arr * (1.0 - offset - tail)

    # --- Locate bins where the CDF crosses each target -----------------------
    # searchsorted finds the first bin where CDF >= target
    idx = np.searchsorted(cdf, targets, side="left")
    idx = np.clip(idx, 0, len(masses) - 1)

    # CDF just below and at the selected bin
    cdf_lo = np.concatenate(([0.0], cdf[:-1]))[idx]
    cdf_hi = cdf[idx]

    # Corresponding bin edges
    x_lo = edges[idx]
    x_hi = edges[idx + 1]

    # --- Interpolate inside the bin -----------------------------------------
    # Assume uniform distribution within the bin
    t = np.where(
        cdf_hi > cdf_lo,
        (targets - cdf_lo) / (cdf_hi - cdf_lo),
        0.0,
    )

    # Final quantile values
    return x_lo + t * (x_hi - x_lo)
