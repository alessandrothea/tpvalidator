from typing import Tuple, Optional, Union, Sequence, Dict
import numpy as np
import pandas as pd


def compute_histogram_ratio(
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


def uproot_hist_mean_std(h):
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
    

def calculate_natural_bins( series : pd.Series, downsampling=10 ):

    x_min=series.min()
    x_max=series.max()

    x_range=(x_max-x_min)
    n_bins=int(x_range)//downsampling
    dx = x_range/n_bins

    bins = [ (x_min + i*dx) for i in range(n_bins+1)]

    return bins
