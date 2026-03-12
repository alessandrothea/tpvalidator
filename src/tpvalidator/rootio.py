import logging
import uproot
import awkward as ak
import pandas as pd
import numpy as np
import json
import os
from typing import Optional, Union, Dict

_log = logging.getLogger(__name__)


def _is_readable_file(file_path: str) -> bool:
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)


def read_tree(tree, cut: Optional[str] = None, branch_names: Optional[list] = None) -> pd.DataFrame:
    """Core primitive: load an uproot TTree into a DataFrame.

    Args:
        tree: open uproot TTree object
        cut (str, optional): uproot cut expression to filter rows. Defaults to None.
        branch_names (list, optional): branches to load. Defaults to None (all branches).

    Returns:
        pd.DataFrame
    """
    arr = tree.arrays(branch_names, library="ak", cut=cut)
    return ak.to_dataframe(arr)


def find_active_channels_branch(tree) -> Optional[str]:
    """Return the name of the active-channels branch in a sparse waveform tree, or None."""
    for name in ['active_channels', 'chans_with_electrons']:
        if name in tree.keys():
            return name
    return None


def read_data(file_path: str, tree_name: str = 'triggerana/tree', branch_names: Optional[list] = None, max_events=None) -> Optional[pd.DataFrame]:
    """Load data from a ROOT tree into a DataFrame after expanding vectors into rows.

    Args:
        file_path (str): path to the ROOT file containing the ROOT tree
        tree_name (str): name or path of the tree in the ROOT file
        branch_names (list, optional): branches to import. Defaults to None (all).
        max_events (int, optional): maximum number of events. Defaults to None.

    Returns:
        pd.DataFrame or None on error.
    """
    if not _is_readable_file(file_path):
        _log.error(f"File is missing or unreadable: {file_path}")
        return None
    try:
        with uproot.open(f'{file_path}:{tree_name}') as tree:
            arr = tree.arrays(branch_names, library="ak", entry_stop=max_events)
            return ak.to_dataframe(arr)
    except Exception as e:
        _log.error(f"Error loading data from {file_path}: {e}")
        return None


def read_info(file_path: str, info_name: str = 'triggerana/info') -> Optional[Dict]:
    """Load processing information from a ROOT file as a Python dictionary.

    Args:
        file_path (str): path to the ROOT file
        info_name (str, optional): path to the TNamed info object. Defaults to 'triggerana/info'.

    Returns:
        dict or None on error.
    """
    if not _is_readable_file(file_path):
        _log.error(f"File is missing or unreadable: {file_path}")
        return None
    try:
        with uproot.open(f'{file_path}:{info_name}') as meta_data:
            return json.loads(meta_data.members['fTitle'])
    except Exception as e:
        _log.error(f"Error loading info from {file_path}: {e}")
        return None


def read_event_list(file_path: str, tree_name: str) -> Optional[pd.DataFrame]:
    """Load the event/run/subrun index from a ROOT tree.

    Args:
        file_path (str): path to the ROOT file
        tree_name (str): name or path of the tree

    Returns:
        pd.DataFrame with columns [event, run, subrun], or None on error.
    """
    if not _is_readable_file(file_path):
        _log.error(f"File is missing or unreadable: {file_path}")
        return None
    try:
        with uproot.open(f'{file_path}:{tree_name}') as tree:
            return tree.arrays(["event", "run", "subrun"], library='pd')
    except Exception as e:
        _log.error(f"Error loading event list from {file_path}: {e}")
        return None


def read_sparse_waveforms(file_path: str, tree_name: str = 'triggerana/rawdigis_tree', ev_sel: Union[int, list] = 1) -> Optional[pd.DataFrame]:
    """Load sparse rawdigits waveforms for a specific event from a ROOT file.

    Only channels listed in the active-channels branch are loaded. Currently
    only the first event (ev_sel=1) is supported.

    Args:
        file_path (str): path to the ROOT file
        tree_name (str, optional): name of the tree. Defaults to 'triggerana/rawdigis_tree'.
        ev_sel: event selection (only first event currently supported).

    Returns:
        pd.DataFrame with columns [event, run, subrun, <channel_ids>, sample_id], or None on error.
    """
    if not _is_readable_file(file_path):
        _log.error(f"File is missing or unreadable: {file_path}")
        return None
    try:
        with uproot.open(f'{file_path}:{tree_name}') as tree:

            activ_chans_branch = find_active_channels_branch(tree)
            if activ_chans_branch is None:
                raise RuntimeError(
                    "Active channel branch not found in tree. "
                    "This doesn't look like a sparse waveform tree."
                )

            branches = ["event", "run", "subrun", activ_chans_branch]
            df_evs = tree.arrays(branches, library='np')
            df_evs = pd.DataFrame(df_evs)

            if not (type(ev_sel) == int and ev_sel == 1):
                raise RuntimeError("Only the loading of the first event is supported")

            ev_num = df_evs.event[0]
            chans = list(df_evs[df_evs.event == ev_num][activ_chans_branch][0])
            _log.debug(f"Found {len(chans)} active channels")

            _log.debug("Loading tree into numpy arrays")
            arrays = tree.arrays(["event", "run", "subrun"] + [str(c) for c in chans], library='np')
            _log.debug("Converting to DataFrame")
            df = pd.DataFrame(arrays)

            df.columns = [int(c) if c not in ["event", "run", "subrun"] else c for c in df.columns]

            _log.debug("Expanding waveforms")
            df_waveforms = df.explode(chans)
            df_waveforms = df_waveforms.astype({c: 'uint16' for c in chans})
            df_waveforms['sample_id'] = np.arange(0, len(df_waveforms))

            return df_waveforms

    except Exception as e:
        _log.error(f"Error loading sparse waveform data from {file_path}: {e}")
        return None


def read_waveforms(filepath: str, channel_ids: list, tree_name: str = 'triggerana/rawdigis_tree', max_events: int = 1, first_event: int = 0) -> Optional[pd.DataFrame]:
    """Load waveform data for specified channels from a ROOT file.

    Args:
        filepath (str): path to the ROOT file containing waveform data.
        channel_ids (list): list of channel IDs to load waveforms for.
        tree_name (str, optional): name of the tree. Defaults to 'triggerana/rawdigis_tree'.
        max_events (int, optional): maximum number of events to load. Defaults to 1.
        first_event (int, optional): index of the first event to load. Defaults to 0.

    Returns:
        pd.DataFrame or None on error.
    """
    try:
        branch_names = [f"{ch:d}" for ch in channel_ids]
        with uproot.open(f'{filepath}:{tree_name}') as tree:
            arrays = tree.arrays(branch_names, library="ak", entry_stop=max_events)
            df = ak.to_dataframe(arrays)
            df.columns = [int(c) for c in df.columns]
            df.index = np.arange(0, len(df))
            return df
    except Exception as e:
        _log.error(f"Error loading waveform data from {filepath}: {e}")
        return None
