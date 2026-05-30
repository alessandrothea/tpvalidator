
import logging
import uproot
import pandas as pd
import numpy as np

# from rich import print
from typing import Optional, List, Tuple

from .rootio import TriggerNtupleReader, RawWaveformsNtupleReader


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
    MC truth, neutrinos, particles, IDEs, rawadcs).
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
        self._simides_summary_tree_name = f'simides_summary'

        self._rawdigits_tree_name: str = 'rawdigis_tree'

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
        self._simides_summary = None
        self._tps = None

        # RawADCs registry
        self.rawdigis_events = []
        
        self.rawdigits_hists = {}
        self._rawadcs = {}

        # Ancillary information
        self._event_list = None
        self._mctruth_blocks = None

        # Initialize trees
        self._do_init(tps_key)

        self._extra_info.update({
            'num_entries': self.num_entries,
            'event_list': self.event_list,
        })



    ###---- 2g stuff here


    def _do_init(self, tps_key: str):

        self._log.info("Opening Trigger NTuple file")
        self._tuple_rdr = TriggerNtupleReader(self._data_path, analyzer_dir=self._analyzer_name)

        # Extract file handle for direct access to ROOT objects when needed
        f = self._tuple_rdr.file

        self._log.debug(f.keys())
        self._log.info("Adding processing info")
        self.info = self._tuple_rdr.get_info(self._info_name)

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
                ttree = self._tuple_rdr.get_tree(ttree_name)
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
                        self._tps_tree_name = f'{tp_trees_folder.keys(cycle=False)[0]}'
                        self._tps_tree_path = f'{self._tps_folder}/{self._tps_tree_name}'
                        logging.info(f'Loading {self._tps_tree_path}')
                        self.tps_tree = self._tuple_rdr.get_tree(f"{self._tps_tree_path}")
                    case _:
                        raise RuntimeError(f"Found multiple TP keys while expecting one {tp_trees_folder.keys()}")

            self._log.info(f"{self._tps_tree_name} found")
        else:
            self._log.info(f"No {self._tps_folder} folder found")



    @staticmethod
    def _get_event_id_list(tree):
        # TODO: this should return 
        # return tree.arrays(branches=['event', 'run', 'subrun'], library='pd').event.unique()
        return tree.to_df(branches=['event', 'run', 'subrun']).event.unique()



    def _load_dataframe_with_event_cut(self, df_id: str) -> pd.DataFrame:
        """Load dataframe from the selected TTree, applying the event cut for this workspace."""
        tree = getattr(self, f'{df_id}_tree')
        df = TrgDataFrame(tree.to_df(entry_start=self._first_entry, entry_stop=self._last_entry))
        if 'event_uid' not in df.columns:
            df['event_uid'] = df.run.astype("uint64")*1000000+df.subrun.astype("uint64")*100+df.event.astype("uint64")
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
    def simides(self):
        if self._simides is None:
            self._log.debug("Loading IDEs dataset")
            self._simides = self._load_dataframe_with_event_cut('simides')
        return self._simides


    @property
    def simides(self):
        if self._simides_summary is None:
            self._log.debug("Loading IDEs dataset")
            self._simides_summary = self._load_dataframe_with_event_cut('simides')
        return self._simides_summary

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
    def num_entries(self) -> int:
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
        """Add a rawdigits (rawadcs) file to the workspace."""

        self._raw_tuple_rdr = RawWaveformsNtupleReader(data_path)

        self._log.info("Loading rawADC tree")
        self.rawdigits_tree = self._raw_tuple_rdr.get_tree(self._rawdigits_tree_name)

        self.rawdigits_hists = {}
        for k in self._raw_tuple_rdr.keys(cycle=False):
            obj_name = k.split('/')[-1]
            self._log.info("Retrieving rawADC histograms")

            if obj_name.startswith('ADCsPlane') or obj_name.startswith('ADCsNoisePlane'):
                self.rawdigits_hists[obj_name.split(';')[0]] = self._raw_tuple_rdr[k.split(';')[0]]

        # TODO: 
        self._log.info("Load rawdigis event list")
        self.rawdigis_events = self.rawdigits_tree.event_list()
        self._log.info(f"{len(self.rawdigis_events)} events found")


    def get_rawadcs(self, event, run, subrun, channel_mask: Optional[List[int]] = None) -> pd.DataFrame:
        if self.rawdigis_events.query(f"(event=={event}) & (run=={run}) & (subrun=={subrun})").empty:
            self._log.warning(f"RawADCs for entry ({event}, {run}, {subrun}) are not available")
            return None

        else:
            ev_uid=(event, run, subrun)
            rawadc = self._rawadcs.get(ev_uid, None)
            if rawadc is not None and channel_mask == rawadc.prod_info['channel_mask']:
                return self._rawadcs[ev_uid]

            rwdf = TrgDataFrame(self.rawdigits_tree.to_df(*ev_uid, channel_mask=channel_mask))
            rwdf.prod_info = {
                'channel_mask': channel_mask
            }
            self._rawadcs[ev_uid] = rwdf
            
            return self._rawadcs[ev_uid]

