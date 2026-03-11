import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import contextmanager
from typing import Tuple, Optional, Union, Sequence


@contextmanager
def temporary_log_level(logger, level):
    """Change the logger message level within the context.
    Restore the previous level at context exit.
    """
    old_level = logger.level
    logger.setLevel(level)
    yield
    logger.setLevel(old_level)


@contextmanager
def pandas_backend(backend):
    """Change the pandas graphical backend within the context.
    Restore the previous backend at context exit.
    """
    current_backend = pd.options.plotting.backend
    pd.options.plotting.backend = backend
    yield
    pd.options.plotting.backend = current_backend


def get_hist_layout(n_items, layout=None):
    if layout is not None:
        return layout
    ncols = math.ceil(math.sqrt(n_items))
    nrows = math.ceil(n_items / ncols)
    return (nrows, ncols)


def subplot_autogrid(n_plots, **kwargs):
    n_rows, n_cols = get_hist_layout(n_plots)

    mosaic = []
    i = 0
    for _ in range(n_rows):
        row = []
        for _ in range(n_cols):
            row.append(i if i < n_plots else '.')
            i += 1
        mosaic += [row]

    fig, ax = plt.subplot_mosaic(mosaic, **kwargs)
    return fig, ax


def df_to_tp_rates(df_tp: pd.DataFrame, readout_window: int = None) -> float:
    """
    Calculates the TP rates from the TP dataframe.

    If the drift window is not specified, its length is estimated from TP's min and max `sample_start`.
    The estimate is only reliable for well-populated samples.
    """
    sampling_time = 0.5e-6  # Sampling time 1/2 usec
    readout_window = df_tp.extra_info['readout_window'] if readout_window is None else readout_window

    tot_time = readout_window * sampling_time * len(df_tp.event.unique())

    n_tps = len(df_tp)

    rate = n_tps / (tot_time) if tot_time > 0 else 0
    return rate


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

    Returns:
    bin_centers, ratio, ratio_err, bins
    """
    num_counts, bins = np.histogram(numerator_data, bins=bins, range=range)
    denom_counts, _ = np.histogram(denominator_data, bins=bins, range=range)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.true_divide(num_counts, denom_counts)
        ratio[~np.isfinite(ratio)] = zero_division

        num_err = np.sqrt(num_counts)
        denom_err = np.sqrt(denom_counts)

        safe_num = np.maximum(num_counts, 1)
        safe_denom = np.maximum(denom_counts, 1)

        ratio_err = ratio * np.sqrt(
            (num_err / safe_num)**2 + (denom_err / safe_denom)**2
        )
        ratio_err[~np.isfinite(ratio_err)] = 0

    return bin_centers, ratio, ratio_err, bins
