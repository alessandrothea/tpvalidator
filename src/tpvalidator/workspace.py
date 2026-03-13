
import logging
import uproot
import json
import pandas as pd
import numpy as np
import awkward as ak

from rich import print
from typing import Tuple, Optional, Union, Sequence, Dict

from .rootio import TriggerNtupleReader, find_active_channels_branch, read_sparse_waveforms


class TrgDataFrame(pd.DataFrame):
    # normal properties
    _metadata = ["prod_info", 'extra_info']

    @property
    def _constructor(self):
        return TrgDataFrame

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prod_info = {}
        self.extra_info = {}


class TriggerAnalysisWorkspace:
    """Workspace for loading TA-finder output trees (event_summary, mctruths,
    ta_event_selection, ta_win_cluster_stats, ta_clusters, tps_with_cluster_flags).
    """

    _log = logging.getLogger('TriggerAnalysisWorkspace')

    tree_names = [
        'event_summary',
        'mctruths',
        'ta_event_selection',
        'ta_win_cluster_stats',
        'ta_clusters',
        'tps_with_cluster_flags'
    ]

    def __init__(self, data_path: str, base_folder=''):

        self._base_folder = base_folder
        self._data_path = data_path
        self._event_list = None
        self._trees = {}
        self._dataframes = {}

        with uproot.open(self._data_path) as f:
            for t in self.tree_names:
                self._log.info(f"Adding '{t}' data")
                ttree_name = f"{self._base_folder}/{t}"

                try:
                    ttree = f[ttree_name]
                    self._log.info(f"{ttree_name} found with {ttree.num_entries} rows")
                except uproot.KeyInFileError:
                    self._log.warning(f"Key '{ttree_name}' not found.")
                    ttree = None
                self._trees[t] = ttree

    def __getattr__(self, name):
        if name in self.tree_names:
            return self._get_dataframe(name)
        else:
            raise AttributeError(name)

    def _load_dataframe(self, tree):
        cut = None
        df = TrgDataFrame(tree.arrays(library="np", cut=cut))
        return df

    def _get_dataframe(self, name) -> TrgDataFrame:
        df = self._dataframes.get(name, None)
        if df is None:
            df = self._load_dataframe(self._trees[name])
            self._dataframes[name] = df
        return df


class TriggerPrimitivesWorkspace:
    """Workspace for loading MC production TP files (trigger primitives,
    MC truth, neutrinos, particles, IDEs, waveforms).
    """

    _log = logging.getLogger('TriggerPrimitivesWorkspace')

    # TODO: add arguments to disable truth info loading
    def __init__(self, data_path: str, first_entry: int=None, last_entry: int = None, tps_key : str = None, analyzer_name: str = 'triggerAna', tps_folder: str = 'TriggerPrimitives', extra_info: dict = {}):

        self._tuple_rdr = None

        # Labels and ROOT object paths
        self._analyzer_name = analyzer_name
        self._tps_folder  = tps_folder
        self._info_name = 'info'

        # Standard trees
        self._event_summary_tree_name = f'event_summary'
        self._mctruths_tree_name = f'mctruths'
        self._mcneutrinos_tree_name = f'mcneutrinos'
        self._mcparticles_tree_name = f'mcparticles'
        self._simides_tree_name = f'simides'

        self._rawdigits_tree_name: str = 'triggerana/rawdigis_tree'

        self._data_path = data_path
        self._first_entry = first_entry
        self._last_entry = last_entry

        # Don't forget to copy!
        self._extra_info = extra_info.copy()

        # Dataframes
        self._event_summary = None
        self._mctruths = None
        self._mcneutrinos = None
        self._mcparticles = None
        self._simides = None
        self._tps = None

        # Waveforms registry
        self.rawdigis_events = []
        self._waveforms = {}

        # Ancillary information
        self._event_list = None
        self._mctruth_blocks = None

        # Initialize trees
        self._do_init_2g(tps_key)

        self._extra_info.update({
            'num_events': self.num_events,
            'event_list': self.event_list,
        })



    ###---- 2g stuff here


    def _do_init_2g(self, tps_key: str):

        self._log.info("Opening Trigger NTuple file")
        self._tuple_rdr = TriggerNtupleReader(self._data_path, analyzer_dir=self._analyzer_name)

        # Extract file handle for direct access to ROOT objects when needed
        f = self._tuple_rdr.file

        self._log.debug(f.keys())
        self._log.info("Adding processing info")
        self.info = self._tuple_rdr.read_info(self._info_name)

        tree_names = [
            'event_summary',
            'mctruths',
            'mcneutrinos',
            'mcparticles',
            'simides'
        ]

        for t in tree_names:
            self._log.info(f"Adding '{t}' data")
            ttree_name = getattr(self, f'_{t}_tree_name')

            try:
                ttree = self._tuple_rdr.read_tree(ttree_name)
            except uproot.KeyInFileError:
                self._log.warning(f"Key '{ttree_name}' not found in file.")
                ttree = None
            setattr(self, f'{t}_tree', ttree)

        # Add trigger primitives
        if self._tps_folder in f[f'{self._analyzer_name}']:

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
                        self._tps_tree_name = f'{self._tps_folder}/{tp_trees_folder.keys(cycle=False)[0]}'
                        logging.info(f'Loading {self._tps_tree_name}')
                        self.tps_tree = self._tuple_rdr.read_tree(f"{self._tps_tree_name}")
                    case _:
                        raise RuntimeError(f"Found multiple TP keys while expecting one {tp_trees_folder.keys()}")

            self._log.info(f"{self._tps_tree_name} found")
            print(f"{self._tps_tree_name} found")
        else:
            self._log.info(f"No {self._tps_folder} folder found")

    ###


    # def _do_init(self, tps_key: str):

    #     self._log.info("Opening Trigger NTuple file")

    #     with uproot.open(self._data_path) as f:

    #         self._log.debug(f.keys())
    #         self._log.info("Adding processing info")
    #         self.info = self._read_infos(f, self._info_name)

    #         tree_names = [
    #             'event_summary',
    #             'mctruths',
    #             'mcneutrinos',
    #             'mcparticles',
    #             'simides'
    #         ]

    #         for t in tree_names:
    #             self._log.info(f"Adding '{t}' data")
    #             ttree_name = getattr(self, f'_{t}_tree_name')

    #             try:
    #                 ttree = f[ttree_name]
    #                 self._log.info(f"{ttree_name} found with {ttree.num_entries} rows")
    #             except uproot.KeyInFileError:
    #                 self._log.warning(f"Key '{ttree_name}' not found in file.")
    #                 ttree = None
    #             setattr(self, f'{t}_tree', ttree)

    #         # Add trigger primitives
    #         if self._tps_folder in f[f'{self._analyzer_name}']:

    #             tp_trees_folder = f[f'{self._analyzer_name}/{self._tps_folder}']
    #             if tps_key:
    #                 logging.info(f'Loading {tps_key}')
    #                 self.tps_tree = tp_trees_folder[tps_key]
    #                 self._tps_tree_name = tps_key
    #             else:
    #                 match len(tp_trees_folder.keys(cycle=False)):
    #                     case 0:
    #                         self.tps_tree = None
    #                     case 1:
    #                         self._tps_tree_name = tp_trees_folder.keys(cycle=False)[0]
    #                         logging.info(f'Loading {self._tps_tree_name}')
    #                         self.tps_tree = tp_trees_folder[self._tps_tree_name]
    #                     case _:
    #                         raise RuntimeError(f"Found multiple TP keys while expecting one {tp_trees_folder.keys()}")

    #             self._log.info(f"{self._tps_tree_name} found with {self.tps_tree.num_entries} rows")
    #         else:
    #             self._log.info(f"No {self._tps_folder} folder found")


    # def _read_infos(self, tfile, info_path) -> Dict:
    #     named_info = tfile[info_path]
    #     return json.loads(named_info.members['fTitle'])


    @staticmethod
    def _get_event_id_list(tree):
        return tree.arrays(['event'], library='pd').event.unique()


    def _load_dataframe_with_event_cut(self, df_id: str) -> pd.DataFrame:
        """Load dataframe from the selected TTree, applying the event cut for this workspace."""
        tree = getattr(self, f'{df_id}_tree')
        df = TrgDataFrame(tree.to_df(entry_start=self._first_entry, entry_stop=self._last_entry))

        df.prod_info = self.info
        df.extra_info = self._extra_info
        return df


    # FIXME:
    def _decorate_tps_dataframe(self):
        """Decorate TPS dataframe with extra columns useful for analysis:
        time_peak, sample_start, sample_peak, bt_is_signal.
        """
        self.tps['time_peak'] = self.tps.time_start+self.tps.samples_to_peak*32
        self.tps['sample_start'] = self.tps.time_start//32
        self.tps['sample_peak'] = self.tps.sample_start+self.tps.samples_to_peak
        self.tps['bt_is_signal'] = (self.tps.bt_numelectrons > 0).astype(np.int8)


    @property
    def tp_maker_name(self):
        return self._tps_tree_name.replace('_', ':')


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
        if self._simides is None:
            self._log.debug("Loading IDEs dataset")
            self._simides = self._load_dataframe_with_event_cut('ides')
        return self._simides


    @property
    def tps(self):
        if self._tps is None:
            self._log.debug("Loading tps dataset")
            self._tps = self._load_dataframe_with_event_cut('tps')
            self._decorate_tps_dataframe()
        return self._tps


    @property
    def mctruth_blocks_map(self):
        if self._mctruth_blocks is None:


            if 'mctruth_blockid_map' in self.info:
                self._mctruth_blocks = dict(self.info['mctruth_blockid_map'])
            else:
                self._mctruth_blocks = dict(
                    self.mctruths[["block_id", "generator_name"]].drop_duplicates().values
                )
        return self._mctruth_blocks

    #
    # Workspace properties
    #
    @property
    def num_events(self) -> int:
        """Number of events in the workspace, based on the event_summary tree."""
        return len(self.event_list)

    @property
    def event_list(self) -> pd.DataFrame:
        """DataFrame of events in the workspace, with columns ``event``, ``run``, and ``subrun``.

        Lazily loaded from the ``event_summary`` tree on first access. Respects the
        ``first_entry`` / ``last_entry`` slice configured at construction time.
        """
        if self._event_list is None:
            self._event_list = self.event_summary_tree.to_df(branches=['event', 'run', 'subrun'], entry_start=self._first_entry, entry_stop=self._last_entry)
        return self._event_list


    #----------------------------------------------------------------------------------------------------------
    # Support for raw dataforms loading
    #
    def add_rawdigits(self, data_path: str):
        """Add a rawdigits (waveforms) file to the workspace."""
        self._rawdigits_path = data_path
        with uproot.open(self._rawdigits_path) as f:
            self._log.info("Loading rawADC tree")
            self.rawdigits_tree = f[self._rawdigits_tree_name]

            self.rawdigits_hists = {}
            for k in f.keys(cycle=False):
                obj_name = k.split('/')[-1]
                self._log.info("Retrieving rawADC histograms")

                if obj_name.startswith('ADCsPlane') or obj_name.startswith('ADCsNoisePlane'):
                    self.rawdigits_hists[obj_name.split(';')[0]] = f[k.split(';')[0]]

        self._load_rawdigis_event_list()


    def get_waveforms(self, ev: int) -> pd.DataFrame:
        if not ev in self.rawdigis_events:
            self._log.warn(f"Waveforms for event {ev} are not available")
            return None

        else:
            df_wf = self._waveforms.get(ev, None)

            if not df_wf is None:
                return df_wf

            if (self._find_rawdigit_tree_active_channels_branch()):
                df_wf = self._load_sparse_waveform_data(ev)
            else:
                df_wf = self._load_waveform_data(ev)

            self._waveforms[ev] = df_wf
            return self._waveforms[ev]


    def _load_rawdigis_event_list(self):
        self._log.info("Load rawdigis event list")
        self.rawdigis_events = self._get_event_id_list(self.rawdigits_tree)
        self._log.info(f"{len(self.rawdigis_events)} events found")


    def _load_waveform_data(self, ev_sel: Union[int, list] = 1):
        """Load waveform data for specified channels from a ROOT file into a pandas DataFrame."""
        try:
            with uproot.open(f'{self._rawdigits_path}:{self._rawdigits_tree_name}') as tree:

                branches = ["event", "run", "subrun"]

                df_evs = tree.arrays(branches, library='pd')

                if not (type(ev_sel) == int and ev_sel == 1):
                    raise RuntimeError("Only the loading of the first event is supported")

                ev_num = df_evs.event[0]
                chans = [o.name for o in tree.branches if o.name not in ['event', 'run', 'subrun']]
                self._log.debug(f"found {len(chans)} channels")

                self._log.debug("Loading tree into np arrays")
                arrays = tree.arrays(library='np')
                self._log.debug("Done loading tree into np arrays")

                self._log.debug("Converting np arrays to dataframe")
                df = pd.DataFrame(arrays)
                self._log.debug("Done converting np arrays to dataframe")

                df.columns = [int(c) if c not in ["event", "run", "subrun"] else c for c in df.columns]

                self._log.debug("Expanding waveforms")
                df_waveforms = df.explode(chans)
                self._log.debug("Done expanding waveforms")

                df_waveforms = df_waveforms.astype({c: 'uint16' for c in chans})
                df_waveforms['sample_id'] = np.arange(0, len(df_waveforms))

                return df_waveforms

        except Exception as e:
            print(f"Error loading sparse waveform data data from {self._rawdigits_path}: {e}")
            return None


    def _find_rawdigit_tree_active_channels_branch(self):
        return find_active_channels_branch(self.rawdigits_tree)

    def _load_sparse_waveform_data(self, ev_sel: Union[int, list] = 1):
        """Load sparse rawdigits waveforms for a specific event from a ROOT file."""
        return read_sparse_waveforms(self._rawdigits_path, self._rawdigits_tree_name, ev_sel)
