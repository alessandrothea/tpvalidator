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


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def wrap_phi(a):
    return abs(a)


def calculate_angles(px, py, pz, p_mag, detector_type: str = 'hd'):
    """
    Calculate:
    θ_y (angle w.r.t vertical y-axis --> *should* align with collection plane orientation in hd),
    θ_U (angle w.r.t U-plane wires at +37.5°),
    θ_V (angle w.r.t V-plane wires at -37.5°),
    Based on momentum of the MC particles in x,y,z directions.

    Returns:
    theta_y, theta_U, theta_V, theta_xz, theta_xz_U, theta_xz_V (all in degrees)
    """
    from rich import print

    theta_y = np.degrees(np.arccos(py / p_mag))
    theta_xz = np.degrees(np.arctan2(px, pz))

    match detector_type:
        case 'hd':
            print("Using HD wire geometry (ϑ[y-u]=-37.5°, ϑ[y-v]=37.5°)")
            theta_rot_U = np.radians(-37.5)
            theta_rot_V = np.radians(37.5)
        case 'vd':
            theta_rot_U = np.radians(60)
            theta_rot_V = np.radians(-60)
            print("Using VD strip geometry (ϑ[y-u]=30.0°, ϑ[y-v]=-30.0°)")
        case _:
            raise ValueError(f'detector type {detector_type} not known')

    p_y_U = py * np.cos(theta_rot_U) - pz * np.sin(theta_rot_U)
    p_z_U = py * np.sin(theta_rot_U) + pz * np.cos(theta_rot_U)
    theta_xz_U = np.degrees(np.arctan2(px, p_z_U))
    theta_y_U = 90 - np.degrees(np.arcsin(p_y_U / p_mag))

    p_y_V = py * np.cos(theta_rot_V) - pz * np.sin(theta_rot_V)
    p_z_V = py * np.sin(theta_rot_V) + pz * np.cos(theta_rot_V)
    theta_xz_V = np.degrees(np.arctan2(px, p_z_V))
    theta_y_V = 90 - np.degrees(np.arcsin(p_y_V / p_mag))

    theta_y = 90 - (90 - theta_y).abs()
    theta_xz = 90 - (90 - theta_xz.abs()).abs()
    theta_y_U = 90 - (90 - theta_y_U).abs()
    theta_xz_U = 90 - (90 - theta_xz_U.abs()).abs()
    theta_y_V = 90 - (90 - theta_y_V).abs()
    theta_xz_V = 90 - (90 - theta_xz_V.abs()).abs()

    return theta_y, theta_y_U, theta_y_V, abs(theta_xz), abs(theta_xz_U), abs(theta_xz_V)


def calculate_more_angles(px, py, pz, p_mag):
    """
    Calculate angles between momentum and detector plane directions for VD geometry.

    Returns:
    theta_drift, theta_beam, theta_coll, theta_u, theta_v,
    phi_coll, phi_drift, phi_drift_u, phi_drift_v, phi_ind_u, phi_ind_v (all in degrees)
    """
    angle_ind = np.radians(-30)
    sin_ind = np.sin(angle_ind)
    cos_ind = np.cos(angle_ind)

    e_ind_u = [0, sin_ind, cos_ind]
    k_ind_u = [0, -cos_ind, sin_ind]

    angle_ind = np.radians(30)
    sin_ind = np.sin(angle_ind)
    cos_ind = np.cos(angle_ind)
    e_ind_v = [0, sin_ind, cos_ind]
    k_ind_v = [0, -cos_ind, sin_ind]

    pe_ind_u = e_ind_u[1] * py + e_ind_u[2] * pz
    pk_ind_u = k_ind_u[1] * py + k_ind_u[2] * pz

    pe_ind_v = e_ind_v[1] * py + e_ind_v[2] * pz
    pk_ind_v = k_ind_v[1] * py + k_ind_v[2] * pz

    theta_u = np.degrees(np.arccos(pe_ind_u / p_mag))
    theta_v = np.degrees(np.arccos(pe_ind_v / p_mag))

    phi_coll = wrap_phi(np.degrees(np.arctan2(pz, px)))
    phi_ind_u = wrap_phi(np.degrees(np.arctan2(pk_ind_u, px)))
    phi_ind_v = wrap_phi(np.degrees(np.arctan2(pk_ind_v, px)))

    theta_drift = np.degrees(np.arccos(px / p_mag))
    theta_coll = np.degrees(np.arccos(py / p_mag))
    theta_beam = np.degrees(np.arccos(pz / p_mag))

    phi_drift = np.degrees(np.arctan2(py, pz))
    phi_drift_u = np.degrees(np.arctan2(pe_ind_u, pk_ind_u))
    phi_drift_v = np.degrees(np.arctan2(pe_ind_v, pk_ind_v))

    theta_drift = 90 - (90 - theta_drift).abs()
    phi_drift = 90 - (90 - phi_drift.abs()).abs()
    phi_drift_u = 90 - (90 - phi_drift_u.abs()).abs()
    phi_drift_v = 90 - (90 - phi_drift_v.abs()).abs()

    return theta_drift, theta_beam, theta_coll, theta_u, theta_v, phi_coll, phi_drift, phi_drift_u, phi_drift_v, phi_ind_u, phi_ind_v


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
