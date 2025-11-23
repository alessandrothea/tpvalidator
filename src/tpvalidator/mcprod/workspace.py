
import logging
import uproot
import json
import pandas as pd
import numpy as np

from rich import print
from typing import Tuple, Optional, Union, Sequence, Dict

class TriggerPrimitivesWorkspace:
    
    _log = logging.getLogger('TriggerPrimitivesWorkspace')


    # TODO: add arguments to disable truth info loading
    def __init__(self, data_path: str, first_entry: int=0, last_entry: int = None, tps_key : str = None, analyzer_name: str = 'triggerAna', tps_folder: str = 'TriggerPrimitives'):

        # Labels and ROOT object paths
        self._analyzer_name = analyzer_name
        self._tps_folder  = tps_folder
        self._info_name = f'{self._analyzer_name}/info'
        self._event_summary_tree_name = f'{self._analyzer_name}/event_summary'
        self._mctruths_tree_name = f'{self._analyzer_name}/mctruths'
        self._mcneutrinos_tree_name = f'{self._analyzer_name}/mcneutrinos'
        self._mcparticles_tree_name = f'{self._analyzer_name}/mcparticles'
        self._ides_tree_name = f'{self._analyzer_name}/simides'
        self._rawdigits_tree_name: str = 'triggerana/rawdigis_tree'

        self._data_path = data_path
        self._first_entry = first_entry
        self._last_entry = last_entry

        # Dataframes
        self._event_summary = None
        self._mctruths = None
        self._mcneutrinos = None
        self._mcparticles = None
        self._ides = None
        self._tps = None

        # Waveforms registry
        self.rawdigis_events = []
        self._waveforms = {}

        self.tp_maker_name = None
        self._mctruth_blocks = None

        self._do_init(tps_key)

        # save name of the tpmaker
        self.tp_maker_name = self._tps_tree_name.replace('_',':')



    #
    # Initial implementation 
    #
    def _do_init(self, tps_key:str):

        self._log.info("Opening Trigger Primitives file")
        with uproot.open(self._data_path) as f:
            
            #
            # Add Montecarlo tree objets
            #
            self._log.debug(f.keys())
            self._log.info("Adding processing info")
            self.info = self._read_infos(f, self._info_name) 

            # Try loading the MCTruth tree
            self._log.info("Adding Event Summary data")
            try:
                self.event_summary_tree = f[self._event_summary_tree_name]
                self._log.info(f"{self._event_summary_tree_name} found with {self.event_summary_tree.num_entries} rows")

            except uproot.KeyInFileError:
                self._log.warning(f"Key '{self._event_summary_tree_name}' not found in file.")
                self._event_summary_tree = None

            # Try loading the MCTruth tree
            self._log.info("Adding MCTruth data")
            try:
                self.mctruths_tree = f[self._mctruths_tree_name]
                self._log.info(f"{self._mctruths_tree_name} found with {self.mctruths_tree.num_entries} rows")

            except uproot.KeyInFileError:
                self._log.warning(f"Key '{self._mctruths_tree_name}' not found in file.")
                self.mctruths_tree = None

            # Try loading the MCTruth tree
            self._log.info("Adding MCNeutrino data")
            try:
                self.mcneutrinos_tree = f[self._mcneutrinos_tree_name]
                self._log.info(f"{self._mcneutrinos_tree_name} found with {self.mcneutrinos_tree.num_entries} rows")

            except uproot.KeyInFileError:
                self._log.warning(f"Key '{self._mcneutrinos_tree_name}' not found in file.")
                self.mcneutrino_tree = None

            # Try loading the MCTruth tree
            self._log.info("Adding MCParticles data")
            try:
                self.mcparticles_tree = f[self._mcparticles_tree_name]
                self._log.info(f"{self._mcparticles_tree_name} found with {self.mcparticles_tree.num_entries} rows")

            except uproot.KeyInFileError:
                self._log.warning(f"Key '{self._mcparticles_tree_name}' not found in file.")
                self.mcparticles_tree = None


            # Try loading the ides tree
            self._log.info("Adding IDEs data")
            try:
                self.ides_tree = f[self._ides_tree_name]
                self._log.info(f"{self._ides_tree_name} found with {self.ides_tree.num_entries} rows")

            except uproot.KeyInFileError:
                self._log.warning(f"Key '{self._ides_tree_name}' not found in file.")
                self.ides_tree = None


            # Add trigger primitives
            tp_trees_folder = f[f'{self._analyzer_name}/{self._tps_folder}']
            if tps_key:
                logging.info(f'Loading {tps_key}')
                self.tps_tree = tp_trees_folder[tps_key]
                self._tps_tree_name = tps_key
            else:

                match len(tp_trees_folder.keys(cycle=False)):
                    case 0:
                        self.tps_tree = None
                    case 1:
                        self._tps_tree_name = tp_trees_folder.keys(cycle=False)[0]
                        logging.info(f'Loading {self._tps_tree_name}')
                        self.tps_tree = tp_trees_folder[self._tps_tree_name]
                    case _:
                        raise RuntimeError(f"Found multiple TP keys while expecting one {tp_trees_folder.keys()}")
                
            self._log.info(f"{self._tps_tree_name} found with {self.tps_tree.num_entries} rows")


    def _read_infos(self, tfile, info_path) -> Dict:
        named_info  = tfile[info_path]
        return json.loads(named_info.members['fTitle'])

    def _events(self, tree):
        return tree.arrays(['event'], library='pd').event.unique()

    
    def _load_dataframe_with_event_cut(self, df_id: str) -> pd.DataFrame:
        """Load dataframe from the selected TTree, applying the event cut for this workspace

        Args:
            df_id (std): id of the dataframe in the workspace

        Returns:
            pd.DataFrame: Dataframe filled with TTree entries
        """
        tree = getattr(self, f'{df_id}_tree')
        ev_cut = self.get_event_selection_str(tree) if tree.num_entries > 0 else None
        self._log.debug(f"Applying event cut to {df_id}")
        return tree.arrays(library="pd", cut=ev_cut)

    # FIXME:
    def _decorate_tps(self):
        """Decorate TPS dataframe with extra columns useful for analysis

        - time_peak: time of the TP peak in DTS units
        - samples_start: the TP start time in sample unites
        - samples_peak: the TP peak time in samples unit

        """
        self.tps['time_peak'] = self.tps.time_start+self.tps.samples_over_threshold*32
        self.tps['sample_start'] = self.tps.time_start//32
        self.tps['sample_peak'] = self.tps.sample_start+self.tps.samples_over_threshold
        self.tps['bt_is_signal'] = self.tps.bt_numelectrons > 0


    def get_event_selection_str(self, tree) -> str:
        """Returns the event selection string based on the first/last entry fields

        Args:
            tree (uproot.Tree): ROOT Tree the selection will be applied to

        Returns:
            str: query selection string
        """

        ev_list = self._events(tree)

        cuts = []
        if not self._first_entry is None:
            # First entry goes from 0 t N, positive
            cuts.append(f'(event >= {ev_list[self._first_entry]})')
        
        if not self._last_entry is None:
            cuts.append(f'(event <= {ev_list[self._last_entry]})')

        event_cut =  ' & '.join(cuts) if len(cuts) > 0 else None
        return event_cut

    # tree getters
    @property
    def event_summary(self):
        if self._event_summary is None:
            self._log.info("Loading event summary dataset")
            self._event_summary = self._load_dataframe_with_event_cut('event_summary')
        return self._event_summary

    @property
    def mctruths(self):
        if self._mctruths is None:
            self._log.debug("Loading MCTruth dataset")
            self._mctruths = self._load_dataframe_with_event_cut('mctruths')

        return self._mctruths

    @property
    def mcneutrinos(self):
        if self._mcneutrinos is None:
            self._log.debug("Loading MCNeutrino dataset")
            self._mcneutrinos = self._load_dataframe_with_event_cut('mcneutrinos')
        return self._mcneutrinos
    
    @property
    def mcparticles(self):
        if self._mcparticles is None:
            self._log.debug("Loading MCParticles dataset")
            self._mcparticles = self._load_dataframe_with_event_cut('mcparticles')

        return self._mcparticles
    
    @property
    def ides(self):
        if self._ides is None:
            self._log.debug("Loading IDEs dataset")
            self._ides = self._load_dataframe_with_event_cut('ides')
        return self._ides

    @property
    def tps(self):
        if self._tps is None:
            self._log.debug("Loading tps dataset")
            self._tps = self._load_dataframe_with_event_cut('tps')
            self._decorate_tps()
        return self._tps
    

    @property
    def mctruth_blocks_map(self):
        if self._mctruth_blocks is None:
            self._mctruth_blocks = dict(
                self.mctruths[["block_id", "generator_name"]].drop_duplicates().values
            )
        return self._mctruth_block
    
    #
    # Workspace properties
    # 
    @property
    def num_events(self) -> int :
        """Number of events in the workspace, based on the event_summary tree

        Returns:
            int: number of events in the workspace
        """
        return len(self.event_summary.groupby(by=['event', 'run', 'subrun']))


    #
    # Support for raw dataforms loading
    #
    def add_rawdigits(self, data_path: str):
        """Add a rawdigits (waveforms) file to the workspace

        Args:
            data_path (str): _description_
        """

        self._rawdigits_path = data_path
        with uproot.open(self._rawdigits_path) as f:
            # Import the 
            self._log.info("Loading rawADC tree")
            self.rawdigits_tree = f[self._rawdigits_tree_name]
        
            # Read 
            self.rawdigits_hists = {}
            for k in f.keys(cycle=False):
                obj_name = k.split('/')[-1]
                self._log.info("Retrieving rawADC histograms")

                if obj_name.startswith('ADCsPlane') or obj_name.startswith('ADCsNoisePlane'):
                    self.rawdigits_hists[obj_name.split(';')[0]] = f[k.split(';')[0]]

        self._load_rawdigis_event_list();


    def get_waveforms(self, ev: int) -> pd.DataFrame:
        if not ev in self.rawdigis_events:
            self._log.warn(f"Waveforms for event {ev} are not available")
            return None
        
        else:
            df_wf = self._waveforms.get(ev, None)

            if not df_wf is None:
                return df_wf
            
            if ( self._find_rawdigit_tree_active_channels_branch() ):
                df_wf = self._load_sparse_waveform_data(ev)
            else:
                df_wf = self._load_waveform_data(ev)

            self._waveforms[ev] = df_wf

            return self._waveforms[ev]


    def _load_rawdigis_event_list( self ):
        self._log.info("Load rawdigis event list")
        self.rawdigis_events = self._events(self.rawdigits_tree)
        self._log.info(f"{len(self.rawdigis_events)} events found")


    def _load_waveform_data(self, ev_sel: Union[int, list] = 1):
        """
        Load waveform data for specified channels from a ROOT file into a pandas DataFrame.

        Args:

        Returns:
            pd.DataFrame or None: DataFrame containing the waveform data for the specified channels, or None on error.
        """
        try:
            with uproot.open(f'{self._rawdigits_path}:{self._rawdigits_tree_name}') as tree:

                branches = ["event", "run", "subrun"]
                
                df_evs = tree.arrays(branches, library='pd')
                print(df_evs.event.values)
                print(ev_sel)

                if not (type(ev_sel) == int and ev_sel == 1):
                    raise RuntimeError("Only the loading of the first event is supported")
                
                ev_num = df_evs.event[0]
                chans = [ o.name for o in tree.branches if o.name not in ['event','run', 'subrun']]
                self._log.debug(f"found {len(chans)} channels")

                self._log.debug("Loading tree into np arrays")
                # arrays = tree.arrays(["event", "run", "subrun"]+[str(c) for c in chans], library='np')
                arrays = tree.arrays( library='np')
                self._log.debug("Done loading tree into np arrays")

                self._log.debug("Converting np arrays to dataframe")
                df = pd.DataFrame(arrays)
                self._log.debug("Done converting np arrays to dataframe")

                df.columns = [int(c) if c not in ["event", "run", "subrun"] else c for c in df.columns]

                self._log.debug("Expanding waveforms")
                df_waveforms = df.explode(chans)
                self._log.debug("Done expanding waveforms")

                # FIXME: this causes a fragmentation warning
                # Try: new_cols = {c: df[c].astype("uint16") for c in chans}
                # df = df.assign(**new_cols)  # single, consolidated assignment
                df_waveforms = df_waveforms.astype({c:'uint16' for c in chans})
                df_waveforms['sample_id'] = np.arange(0, len(df_waveforms))

                return df_waveforms
                
            # branch_names = [f"{ch:d}" for ch in channel_ids ]
            # with uproot.open(f'{filepath}:{tree_name}') as tree:
            #     arrays = tree.arrays(branch_names, library="ak", entry_stop=max_events)
            #     df = ak.to_dataframe(arrays)
            #     df.columns = [int(c) for c in df.columns]
            #     df.index = np.arange(0, len(df))
            #     return df
            
        except Exception as e:
            print(f"Error loading sparse waveform data data from {self._rawdigits_path}: {e}")
            return None


    def _find_rawdigit_tree_active_channels_branch(self):

        try:
            with uproot.open(f'{self._rawdigits_path}:{self._rawdigits_tree_name}') as tree:

                branch_names = tree.keys()
                
                for name in ['active_channels', 'chans_with_electrons']:
                    if name in branch_names:
                        return name
                return None
        except Exception as e:
            print(f"Error loading sparse waveform data data from {self._rawdigits_path}: {e}")
            return None

    def _load_sparse_waveform_data(self, ev_sel: Union[int, list] = 1):
        """Loads sparse rawdigits waveforms for a specific event from a ROOT file.

        Args:
            ev_num (int): Event number to load waveforms for.
            file_path (str): Path to the ROOT file containing the data.
            tree_name (str, optional): Name of the tree in the ROOT file. Defaults to 'triggerana/rawdigis_tree'.

        Returns:
            pd.DataFrame or None: DataFrame containing the waveforms for the specified event, or None if not found or on error.

        """



        try:
            with uproot.open(f'{self._rawdigits_path}:{self._rawdigits_tree_name}') as tree:

                activ_chans_branch = self._find_rawdigit_tree_active_channels_branch(tree)
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


                # extract the list of channels with signal from the 'chans_with_electrons' branch
                chans = ([ c for c in df_evs[df_evs.event == ev_num][activ_chans_branch][0]])
                
                # arrays = tree.arrays(["event", "run", "subrun"]+[str(c) for c in chans])
                # df_waveforms = ak.to_dataframe(arrays)
                # df_waveforms.columns = [int(c) if c not in ["event", "run", "subrun"] else c for c in df_waveforms.columns]

                # df_waveforms['sample_id'] = np.arange(0, len(df_waveforms))
                # return df_waveforms
                self._log.debug(f"found {len(chans)} channels")

                self._log.debug("Loading tree into np arrays")
                arrays = tree.arrays(["event", "run", "subrun"]+[str(c) for c in chans], library='np')
                self._log.debug("Done loading tree into np arrays")

                self._log.debug("Converting np arrays to dataframe")
                df = pd.DataFrame(arrays)
                self._log.debug("Done converting np arrays to dataframe")

                df.columns = [int(c) if c not in ["event", "run", "subrun"] else c for c in df.columns]

                self._log.debug("Expanding waveforms")
                df_waveforms = df.explode(chans)
                self._log.debug("Done expanding waveforms")

                # FIXME: this causes a fragmentation warning
                # Try: new_cols = {c: df[c].astype("uint16") for c in chans}
                # df = df.assign(**new_cols)  # single, consolidated assignment
                df_waveforms = df_waveforms.astype({c:'uint16' for c in chans})
                df_waveforms['sample_id'] = np.arange(0, len(df_waveforms))

                return df_waveforms
        except Exception as e:
            print(f"Error loading sparse waveform data data from {self._rawdigits_path}: {e}")
            return None