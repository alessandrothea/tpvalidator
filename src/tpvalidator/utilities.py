import uproot
import awkward as ak
import pandas as pd
import numpy as np
import json
import math
from typing import Tuple, Optional, Union, Sequence, Dict
from rich import print

from contextlib import contextmanager

#
# True utilities
#
@contextmanager
def temporary_log_level(logger, level):
    """Change the logger message lever within the context.
    Restore the previous leve at context exit.
    """
    old_level = logger.level
    logger.setLevel(level)
    yield
    logger.setLevel(old_level)


@contextmanager
def pandas_backend(backend):
    """Change the pandas graphical backend within the context.
    Restore the previous backend at context exit.

    Args:
        backend (_type_): _description_
    """
    import pandas as pd

    current_backend = pd.options.plotting.backend
    pd.options.plotting.backend = backend
    yield
    pd.options.plotting.backend = current_backend

##
#
# TODO: Cleanup
# Outdated - possibly replaced by workspace
#
##

_tpgtree_folder_name = 'triggerana'
_tpgtree_tp_tree_name = 'tree'
_tpgtree_charge_tree_name = 'q_tree'
_tpgtree_rawdigits_tree_name = 'rawdigis_tree'

def load_data(file_path: str, tree_name: str = 'triggerana/tree', branch_names: Optional[list] = None, max_events=None) -> pd.DataFrame:
    """
    Loads data from a ROOT tree into a Pandas Dataframe after expanding vectors into rows.

    Args:
        file_path (str): path to the ROOT file containg the ROOT tree
        tree_name (str): name or path of the tree in the ROOT file
        branch_names (list): branches to import in the dataframe. 
        max_events (int, optional): maximum number of events. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the expanded tree data with vectors expanded into rows.
    """
    try:
        with uproot.open(f'{file_path}:{tree_name}') as tree:
            arrays = tree.arrays(branch_names, library="ak", entry_stop=max_events)
            return ak.to_dataframe(arrays)
    
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None
    

def load_info(file_path: str, info_name: str = 'triggerana/info') -> Dict:
    """Laod processing information from tpgtree file as a python dictionary

    Args:
        file_path (str): path to the root file
        info_name (str, optional): . Defaults to 'triggerana/info'.

    Returns:
        dict: Dictionary containing tpg processing information
    """
    try:
        with uproot.open(f'{file_path}:{info_name}') as meta_data:
            json_data = meta_data.members['fTitle']
            return json.loads(json_data)
    
    except Exception as e:
        print(f"Error loading info from {file_path}: {e}")
        return None
    


def load_event_list(file_path: str, tree_name: str):
    try:
        with uproot.open(f'{file_path}:{tree_name}') as tree:
            branches = ["event", "run", "subrun"]
            df_evs = tree.arrays(branches, library='pd')
            return df_evs
    except Exception as e:
        print(f"Error loading sparse waveform data data from {file_path}: {e}")
        return None


def load_sparse_waveform_data(file_path: str, tree_name: str = 'triggerana/rawdigis_tree', ev_sel: Union[int, list] = 1):
    """Loads sparse rawdigits waveforms for a specific event from a ROOT file.

    Args:
        ev_num (int): Event number to load waveforms for.
        file_path (str): Path to the ROOT file containing the data.
        tree_name (str, optional): Name of the tree in the ROOT file. Defaults to 'triggerana/rawdigis_tree'.

    Returns:
        pd.DataFrame or None: DataFrame containing the waveforms for the specified event, or None if not found or on error.

    """

    def find_active_channels_branch(tree):

        branch_names = tree.keys()
        
        for name in ['active_channels', 'chans_with_electrons']:
            if name in branch_names:
                return name
        return None

    try:
        with uproot.open(f'{file_path}:{tree_name}') as tree:

            activ_chans_branch = find_active_channels_branch(tree)
            if activ_chans_branch is None:
                raise RuntimeError(f"Active channel branch not found in tree. This doesn't look like a sparse waveform tree")
            branches = ["event", "run", "subrun"]+[activ_chans_branch]
            
            df_evs = tree.arrays(branches, library='pd')

            print(df_evs.event.values)
            print(ev_sel)

            if not (type(ev_sel) == int and ev_sel == 1):
                raise RuntimeError("Only the loading of the first event is supported")
            

            # # TODO: support list of events
            # if not (df_evs.event == ev_sel).any():
            #     # Event not present in the list
            #     return None
            
            ev_num = df_evs.event[0]


            # extract the list of channels with stingal from the 'chans_with_electrons' branch
            chans = ([ c for c in df_evs[df_evs.event == ev_num][activ_chans_branch][0]])
            print(f"found {len(chans)} channels")
            # print("Loading akward array")
            # arrays = tree.arrays(["event", "run", "subrun"]+[str(c) for c in chans])
            # print("Done loading akward array")
            # print("Converting akward array to dataframe")
            # df_waveforms = ak.to_dataframe(arrays)
            # print("Done converting akward array to dataframe")

            print("Loading dataframe")
            df_waveforms = tree.arrays(["event", "run", "subrun"]+[str(c) for c in chans], library='pd')
            print("Done loading dataframe")
            # print("Converting akward array to dataframe")
            # df_waveforms = ak.to_dataframe(arrays)
            # print("Done converting akward array to dataframe")

            df_waveforms.columns = [int(c) if c not in ["event", "run", "subrun"] else c for c in df_waveforms.columns]

            df_waveforms['sample_id'] = np.arange(0, len(df_waveforms))
            return df_waveforms
        
    except Exception as e:
        print(f"Error loading sparse waveform data data from {file_path}: {e}")
        return None


def load_waveform_data(filepath, channel_ids, tree_name: str = 'triggerana/rawdigis_tree', max_events=1, first_event=0):
    """
    Load waveform data for specified channels from a ROOT file into a pandas DataFrame.

    Args:
        filepath (str): Path to the ROOT file containing waveform data.
        channel_ids (list): List of channel IDs to load waveforms for.
        tree_name (str, optional): Name of the tree in the ROOT file. Defaults to 'triggerana/rawdigis_tree'.
        max_events (int, optional): Maximum number of events to load. Defaults to 1.
        first_event (int, optional): Index of the first event to load. Defaults to 0.

    Returns:
        pd.DataFrame or None: DataFrame containing the waveform data for the specified channels, or None on error.
    """
    try:
        branch_names = [f"{ch:d}" for ch in channel_ids ]
        with uproot.open(f'{filepath}:{tree_name}') as tree:
            arrays = tree.arrays(branch_names, library="ak", entry_stop=max_events)
            df = ak.to_dataframe(arrays)
            df.columns = [int(c) for c in df.columns]
            df.index = np.arange(0, len(df))
            return df
        
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None


###
# 
# Plotting utilities
#
###

import matplotlib.pyplot as plt

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




###
# FIXME: Move to another file
#
###


def df_to_TP_rates(df_tp: pd.DataFrame) -> int:
    '''
    Calculate the TP rates from the TP dataframe
    '''
    sampling_time = 0.5e-6 # Sampling time 1/2 usec
    tot_samples_est = (df_tp.groupby('event').TP_peakT.max()-df_tp.groupby('event').TP_peakT.min()).sum()
    tot_time_est = tot_samples_est*sampling_time
    n_tps = len(df_tp)

    # print(f"Integrated number of samples in the dataset (over all events): {tot_samples_est}")
    # print(f"Integrated simulated time: {tot_samples_est*sampling_time} s")

    # print(f"Integrated number of TPs: {n_tps}")

    return n_tps/(tot_time_est) if tot_time_est > 0 else 0


def calculate_angles(px, py, pz, p_mag, detector_type: str = 'hd'):
    """
    Calculate:
    θ_y (angle w.r.t vertical y-axis --> *should* align with collection plane orientation in hd),
    θ_U (angle w.r.t U-plane wires at +37.5°),
    θ_V (angle w.r.t V-plane wires at -37.5°),
    Based on momentum of the MC particles in x,y,z directions 
        
    Returns:
    theta_y, theta_U, theta_V, theta_xz, theta_xz_U, theta_xz_V (all in degrees)
    """
    # θ_y: Angle relative to vertical y-axis, where θ_y = 0 is upward
    #theta_y = np.degrees( np.arcsin(py / p_mag))
    theta_y = np.degrees(np.arccos(py / p_mag))
    #theta_y = (theta_y + 180) % 360 - 180
    
    # θ_xz: Angle in the xz-plane (measured from z-axis)
    theta_xz = np.degrees(np.arctan2(px, pz))
    
    # Rotation angles for U and V planes (±37.5 degrees in zy-plane)
    match detector_type:
        case 'hd':
            print("Using HD wire geometry (ϑ[y-u]=-37.5°, ϑ[y-v]=37.5°)")
            theta_rot_U = np.radians(-37.5)  # U-plane rotation
            theta_rot_V = np.radians(37.5)   # V-plane rotation
        # case 'vd':
            # print("Using VD strip geometry (ϑ[y-u]=60.0°, ϑ[y-v]=-60.0°)")

            # theta_rot_U = np.radians(60.0)  # U-plane rotation
            # theta_rot_V = np.radians(-60.0)   # V-plane rotation
        case 'vd':

            theta_rot_U = np.radians(60)  # U-plane rotation
            theta_rot_V = np.radians(-60)   # V-plane rotation
            print("Using VD strip geometry (ϑ[y-u]=30.0°, ϑ[y-v]=-30.0°)")

        case _:
            raise ValueError(f'detector type {detector_type} not know')

    
    # Rotate momentum components in zy-plane for U-plane
    p_y_U = py * np.cos(theta_rot_U) - pz * np.sin(theta_rot_U)
    p_z_U = py * np.sin(theta_rot_U) + pz * np.cos(theta_rot_U)
    theta_xz_U = np.degrees(np.arctan2(px, p_z_U))
    theta_y_U =  90 - np.degrees(np.arcsin(p_y_U / p_mag))
    #theta_y_U = (theta_y_U + 180) % 360 - 180 
 
    # Rotate momentum components in zy-plane for V-plane
    p_y_V = py * np.cos(theta_rot_V) - pz * np.sin(theta_rot_V)
    p_z_V = py * np.sin(theta_rot_V) + pz * np.cos(theta_rot_V)
    theta_xz_V = np.degrees(np.arctan2(px, p_z_V))
    theta_y_V = 90-np.degrees(np.arcsin(p_y_V / p_mag))
    #theta_y_V = (theta_y_V + 180) % 360 - 180

    # theta_y = theta_y.where(theta_y < 90, 180-theta_y)
    # theta_y_U = theta_y_U.where(theta_y_U < 90, 180-theta_y_U)
    # theta_y_V = theta_y_V.where(theta_y_V < 90, 180-theta_y_V)


    theta_y = 90-(90-theta_y).abs()
    theta_xz = 90-(90-theta_xz.abs()).abs()
    theta_y_U = 90-(90-theta_y_U).abs()
    theta_xz_U = 90-(90-theta_xz_U.abs()).abs()
    theta_y_V = 90-(90-theta_y_V).abs()
    theta_xz_V = 90-(90-theta_xz_V.abs()).abs()

    return theta_y, theta_y_U, theta_y_V, abs(theta_xz), abs(theta_xz_U), abs(theta_xz_V)

# Enable when ready
#@njit(cache=True, nogil=True)
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def wrap_phi(a):
    conds = [
        a > 90,
        a < -90,
    ]

    choices = [
        a - 180,
        a + 180
    ]
    return abs(a)
    return np.select(conds, choices, default=a)


def calculate_more_angles(px, py, pz, p_mag):
    """
    Calculate:

    1. angle between p and drift direction
    2. angle between p and collection wire direction
    2. angle between p and induction wires direction


    Note about VD geometry
    - z is the direction of the beam
    - The drift direction is upward/downward : x, in vd coordinates
    - y is the direction of collection strips on plane X (hd) or Z (vd)
    - u, the direction of the first induction plane strips is (0, -0.5, 0.866) in LArSoft
    - v, the direction of the first induction plane strips is (0, -0.5, -0.866) in LArSoft


    θ_y (angle w.r.t vertical y-axis --> *should* align with collection plane orientation in hd),
    θ_U (angle w.r.t U-plane wires at +37.5° in hd),
    θ_V (angle w.r.t V-plane wires at -37.5°),
    Based on momentum of the MC particles in x,y,z directions 
        
    Returns:
    theta_drift, theta_beam, theta_coll, theta_u, theta_v, phi_coll, phi_ind_u, phi_ind_v (all in degrees)
    """

    # Induction strips/wires angle wrt z plane
    angle_ind  = np.radians(-30) # degrees
    sin_ind  = np.sin(angle_ind)
    cos_ind  = np.cos(angle_ind)
    # unitary vectors for each plane
    e_coll = [0,1,0]

    # Define induction wires vector basis
    e_ind_u = [0, sin_ind, cos_ind]
    k_ind_u = [0, -cos_ind, sin_ind]

    # e_ind_v = [0, sin_ind, -cos_ind]
    # k_ind_v = [0, cos_ind, sin_ind]

    # angle_ind  = np.radians(-150) # degrees
    angle_ind  = np.radians(30) # degrees
    sin_ind  = np.sin(angle_ind)
    cos_ind  = np.cos(angle_ind)
    e_ind_v = [0, sin_ind, cos_ind]
    k_ind_v = [0, -cos_ind, sin_ind]

    # Calculate prokections on all axes
    pe_ind_u = e_ind_u[1]*py+e_ind_u[2]*pz
    pk_ind_u = k_ind_u[1]*py+k_ind_u[2]*pz

    pe_ind_v = e_ind_v[1]*py+e_ind_v[2]*pz
    pk_ind_v = k_ind_v[1]*py+k_ind_v[2]*pz


    theta_u = np.degrees(np.arccos(pe_ind_u / p_mag))
    theta_v = np.degrees(np.arccos(pe_ind_v / p_mag))

    phi_coll = wrap_phi(np.degrees(np.arctan2 ( pz      , px )))
    phi_ind_u = wrap_phi(np.degrees(np.arctan2( pk_ind_u, px )))
    phi_ind_v = wrap_phi(np.degrees(np.arctan2( pk_ind_v, px )))

    theta_drift = np.degrees(np.arccos(px / p_mag))
    theta_coll = np.degrees(np.arccos(py / p_mag))
    theta_beam = np.degrees(np.arccos(pz / p_mag))

    phi_drift = np.degrees(np.arctan2 ( py      , pz ))
    phi_drift_u = np.degrees(np.arctan2 ( pe_ind_u, pk_ind_u ))
    phi_drift_v = np.degrees(np.arctan2 ( pe_ind_v, pk_ind_v ))

    # phi_drift_u = phi_drift+60
    # phi_drift_u = phi_drift_u.where(phi_drift_u < 240., 360-phi_drift_u).abs()

    theta_drift = 90-(90-theta_drift).abs()
    phi_drift = 90-(90-phi_drift.abs()).abs()
    phi_drift_u = 90-(90-phi_drift_u.abs()).abs()
    phi_drift_v = 90-(90-phi_drift_v.abs()).abs()


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