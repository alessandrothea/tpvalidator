import uproot
import logging
import json
import awkward as ak
import pandas as pd
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
        self._tp_path = data_path

        self._log.info("Opening Trigger Primitives file")
        with uproot.open(self._tp_path) as f:

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



    def add_rawdigits(self, data_path):

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



