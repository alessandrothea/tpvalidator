import uproot
import logging
import json
import awkward as ak
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, Sequence, Dict

class TriggerPrimitivesWorkspace:

    tp_tree_name: str = 'triggerana/tps_tree'
    ide_tree_name: str = 'triggerana/ides_tree'
    mctruth_tree_name: str = 'triggerana/mctruth_tree'
    rawdigits_tree_name: str = 'triggerana/rawdigis_tree'


    MC_BRANCHES : list = ['Eng', 'Ekin', 'startX', 'startY', 'startZ',  'Px', 'Py', 'Pz', 'P']
    BT_BRANCHES : list = ['totQ_X', 'totQ_U', 'totQ_V', 'detQ_X', 'detQ_U', 'detQ_V']
    TP_BRANCHES = [
        'n_TPs', 'TP_channel', 'TP_startT', 'TP_peakT', 'TP_peakADC', 
        'TP_SADC', 'TP_TOT', 'TP_plane', 'TP_TPC', 
        'TP_trueX', 'TP_trueY', 'TP_trueZ', 'TP_true_n_el',
        'TP_signal', 'TP_mcgen_key', 'TP_n_mcgen']

    _log = logging.getLogger('TriggerPrimitivesWorkspace')
    
    def __init__(self, data_path: str):
        self._data_path = data_path

        self._log.info("Opening Trigger Primitives file")
        with uproot.open(self._data_path) as f:

            self._log.info("Retrieving processing info")
            self.info = self._read_infos(f, 'triggerana/info')

            # FIXME: fail if the main tree is not there
            self._log.info("Retrieving Trigger Primitives data")
            self.tree = f[self.tp_tree_name]

            # Try loading the 
            self._log.info("Retrieving IDEs data")
            try:
                self.ides_tree = f[self.ide_tree_name]
                print(self.ides_tree)
            except uproot.KeyInFileError:
                self._log.warning(f"Key '{self.ide_tree_name}' not found in file.")
                self.ides_tree = None

        self._read_tp_dataframes()

        self.waveforms = {}

        
    def __del__(self):
        
        del self.tree
        del self.ides_tree


    # def __repr__(self):
    #     return "TriggerPrimitivesWorkspace"


    def _read_infos(self, tfile, info_path) -> Dict:
        named_info  = tfile[info_path]
        return json.loads(named_info.members['fTitle'])
    

    def _read_tree(self, tree, branch_names : Optional[list] = None, entry_start=None, entry_stop=None) -> pd.DataFrame:
        """Load data from tree and convert it into a pandas dataframe

        Args:
            branch_names (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Load 
        arrays = tree.arrays(branch_names, library="ak", entry_start=entry_start, entry_stop=entry_stop)
        return ak.to_dataframe(arrays)
    

    def _read_tp_dataframes(self) -> None:

        self._log.info("Load event list")
        self.events = self._read_tree(self.tree, branch_names=['event'])
        self._log.info(f"{len(self.events)} events found")
        
        self._log.info("Loading Trigger Primitives data")
        self.tps = self._read_tree(self.tree, branch_names=['event']+self.TP_BRANCHES)

        if not self.ides_tree is None:
            self.ides = self._read_tree(self.ides_tree)

            
    def _load_rawdigis_events( self ):
        self._log.info("Load rawdigis event list")
        self.rawdigis_events = self._read_tree(self.rawdigits_tree, branch_names=['event'])
        self._log.info(f"{len(self.rawdigis_events)} events found")


    def _load_sparse_waveform_data(self, ev_sel: Union[int, list] = 1):
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
            with uproot.open(f'{self._rawdigits_path}:{self.rawdigits_tree_name}') as tree:

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
                # arrays = tree.arrays(["event", "run", "subrun"]+[str(c) for c in chans])
                # df_waveforms = ak.to_dataframe(arrays)
                # df_waveforms.columns = [int(c) if c not in ["event", "run", "subrun"] else c for c in df_waveforms.columns]

                # df_waveforms['sample_id'] = np.arange(0, len(df_waveforms))
                # return df_waveforms
                print(f"found {len(chans)} channels")

                print("Loading tree into np arrays")
                arrays = tree.arrays(["event", "run", "subrun"]+[str(c) for c in chans], library='np')
                print("Done loading tree into np arrays")

                print("Converting np arrays to dataframe")
                df = pd.DataFrame(arrays)
                print("Done converting np arrays to dataframe")

                df.columns = [int(c) if c not in ["event", "run", "subrun"] else c for c in df.columns]

                print("Expanding waveforms")
                df_waveforms = df.explode(chans)
                print("Done expanding waveforms")

                df_waveforms = df_waveforms.astype({c:'uint16' for c in chans})
                df_waveforms['sample_id'] = np.arange(0, len(df_waveforms))

                return df_waveforms
        except Exception as e:
            print(f"Error loading sparse waveform data data from {self._rawdigits_path}: {e}")
            return None


    # def load_waveform_data(filepath, channel_ids, tree_name: str = 'triggerana/rawdigis_tree', max_events=1, first_event=0):
    #     """
    #     Load waveform data for specified channels from a ROOT file into a pandas DataFrame.

    #     Args:
    #         filepath (str): Path to the ROOT file containing waveform data.
    #         channel_ids (list): List of channel IDs to load waveforms for.
    #         tree_name (str, optional): Name of the tree in the ROOT file. Defaults to 'triggerana/rawdigis_tree'.
    #         max_events (int, optional): Maximum number of events to load. Defaults to 1.
    #         first_event (int, optional): Index of the first event to load. Defaults to 0.

    #     Returns:
    #         pd.DataFrame or None: DataFrame containing the waveform data for the specified channels, or None on error.
    #     """
    #     try:
    #         branch_names = [f"{ch:d}" for ch in channel_ids ]
    #         with uproot.open(f'{filepath}:{tree_name}') as tree:
    #             arrays = tree.arrays(branch_names, library="ak", entry_stop=max_events)
    #             df = ak.to_dataframe(arrays)
    #             df.columns = [int(c) for c in df.columns]
    #             df.index = np.arange(0, len(df))
    #             return df
            
    #     except Exception as e:
    #         print(f"Error loading data from {filepath}: {e}")
    #         return None



    def add_rawdigits(self, data_path: str):
        """Add a rawdigits (waveforms) file to the workspace

        Args:
            data_path (str): _description_
        """

        self._rawdigits_path = data_path
        with uproot.open(self._rawdigits_path) as f:
            # Import the 
            self._log.info("Retrieving rawADC tree")
            self.rawdigits_tree = f[self.rawdigits_tree_name]
        
            # Read 
            self.rawdigits_hists = {}
            for k in f.keys():
                obj_name = k.split('/')[-1]
                self._log.info("Retrieving rawADC histograms")

                if obj_name.startswith('ADCsPlane') or obj_name.startswith('ADCsNoisePlane'):
                    self.rawdigits_hists[obj_name.split(';')[0]] = f[k.split(';')[0]]

    
        self._load_rawdigis_events();


    def get_waveforms(self, ev: int) -> pd.DataFrame:
        if not ev in list(self.rawdigis_events.event):
            self._log.warn(f"Waveforms for event {ev} are not available")
            return None
        
        else:
            df_wf = self.waveforms.get(ev, None)

            if not df_wf is None:
                return df_wf
            
            df_wf = self._load_sparse_waveform_data(ev)
            self.waveforms[ev] = df_wf

            return self.waveforms[ev]


    #
    # Convenience workspace info accessors
    #
    def detector_name(self):
        print(self.info)
        if 'geo' in self.info:
            return self.info['geo']['detector']
        elif 'detector' in self.info:
            return self.info['detector']
        else:
            raise KeyError(f"Unable to find detector name in tpg processing info")
        
    def tp_algorithm(self):
        return self.info['tpg']['tool']
    

    def tp_threshold(self, plane: int):
        if plane in [0,1,2]:
            return self.info['tpg'][f'threshold_tpg_plane{plane}']
        else:
            return ValueError(f"Invalid plane id: {plane}")
        
    def tp_backtracker_offset(self, plane: int):
        plane_map = {
            0: 'U_window_offset',
            1: 'V_window_offset',
            2: 'X_window_offset',
        }

        if not plane in plane_map:
            return KeyError(f"Plane '{plane}' not known")
        
        return self.info['tptree'][plane_map[plane]]


