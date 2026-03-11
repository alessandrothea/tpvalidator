import uproot
import awkward as ak
import pandas as pd
import numpy as np
import json
from typing import Optional, Union, Dict
from rich import print


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
    """Load processing information from tpgtree file as a python dictionary.

    Args:
        file_path (str): path to the root file
        info_name (str, optional): Defaults to 'triggerana/info'.

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
        file_path (str): Path to the ROOT file containing the data.
        tree_name (str, optional): Name of the tree in the ROOT file. Defaults to 'triggerana/rawdigis_tree'.
        ev_sel: Event selection (only first event currently supported).

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
                raise RuntimeError("Active channel branch not found in tree. This doesn't look like a sparse waveform tree")
            branches = ["event", "run", "subrun"] + [activ_chans_branch]

            df_evs = tree.arrays(branches, library='pd')

            print(df_evs.event.values)
            print(ev_sel)

            if not (type(ev_sel) == int and ev_sel == 1):
                raise RuntimeError("Only the loading of the first event is supported")

            ev_num = df_evs.event[0]

            chans = ([c for c in df_evs[df_evs.event == ev_num][activ_chans_branch][0]])
            print(f"found {len(chans)} channels")

            print("Loading dataframe")
            df_waveforms = tree.arrays(["event", "run", "subrun"] + [str(c) for c in chans], library='pd')
            print("Done loading dataframe")

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
        branch_names = [f"{ch:d}" for ch in channel_ids]
        with uproot.open(f'{filepath}:{tree_name}') as tree:
            arrays = tree.arrays(branch_names, library="ak", entry_stop=max_events)
            df = ak.to_dataframe(arrays)
            df.columns = [int(c) for c in df.columns]
            df.index = np.arange(0, len(df))
            return df

    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None
