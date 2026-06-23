
import logging
import uproot
import pandas as pd
import numpy as np
from abc import abstractmethod

# from rich import print
from typing import Optional, List, Tuple

from .rootio import TriggerNtupleReader, RawWaveformsNtupleReader


# Helper method to rebuild the entry-subentry multindex, if needed
def rebuild_dataframe_entry_index(df: pd.DataFrame, keys: list[str]):
    # combine columns into a single tuple-key, rank densely to get 0-based entry index
    entry_keys = df[keys].apply(tuple, axis=1)

    df.index = pd.MultiIndex.from_arrays(
        [
            entry_keys.rank(method="dense").astype(int) - 1,  # entry
            df.groupby(entry_keys).cumcount(),                 # subentry
        ],
        names=["entry", "subentry"],
    )

    return df



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
    """Abstract base class for trigger workspaces.

    Provides shared tree/dataframe registry, lazy loading, and attribute-style access.
    Subclasses must define `tree_names` and implement `_load_dataframe(name)`.
    """

    _log = logging.getLogger('TriggerAnalysisWorkspace')
    tree_names: list = []

    def __init__(self):
        self._trees: dict = {}
        self._dataframes: dict = {}
        self._event_list = None
        self._df_decorators: dict = {}

    def __getattr__(self, name):
        if '_trees' in self.__dict__:
            if name.endswith('_tree') and (base := name[:-5]) in self._trees:
                return self._trees[base]
            if '_dataframes' in self.__dict__ and name in self.tree_names:
                return self._get_dataframe(name)
        raise AttributeError(name)

    def get_tree(self, name):
        return self._trees[name]

    def get_df(self, name) -> TrgDataFrame:
        return self._get_dataframe(name)

    def _get_dataframe(self, name) -> TrgDataFrame:
        if name not in self._dataframes:
            self._dataframes[name] = self._load_dataframe(name)
            if name in self._df_decorators:
                self._df_decorators[name]()
        return self._dataframes[name]

    @abstractmethod
    def _load_dataframe(self, name) -> TrgDataFrame:
        ...


class TriggerActivityWorkspace(TriggerAnalysisWorkspace):
    """Workspace for loading TA-finder output trees (event_summary, mctruths,
    ta_event_selection, ta_win_cluster_stats, ta_clusters, tps_with_cluster_flags).
    """

    _log = logging.getLogger('TriggerActivityWorkspace')

    tree_names = [
        'event_summary',
        'mctruths',
        'ta_event_selection',
        'ta_win_stats',
        'ta_win_cluster_stats',
        'ta_clusters',
        'tps_with_cluster_flags'
    ]
    _info_name = 'info'

    def __init__(self, data_path: str, base_folder=''):
        super().__init__()
        self._base_folder = base_folder
        self._data_path = data_path
        self._tuple_rdr = None
        self._do_init()

    def _do_init(self):
        self._log.info("Opening TA-finder output file")
        self._tuple_rdr = TriggerNtupleReader(self._data_path, analyzer_dir=self._base_folder)

        rootfile = self._tuple_rdr.file

        self._log.debug(rootfile.keys())
        self._log.info("Adding processing info")
        self._info = self._tuple_rdr.get_info(self._info_name)
        
        for t in self.tree_names:
            self._log.info(f"Adding '{t}' data")
            try:
                ttree = self._tuple_rdr.get_tree(t)
            except uproot.KeyInFileError:
                self._log.warning(f"Key '{t}' not found in file.")
                ttree = None
            self._trees[t] = ttree

    def _load_dataframe(self, name) -> TrgDataFrame:
        # return TrgDataFrame(self._trees[name].to_df_np())
        df = TrgDataFrame(self._trees[name].to_df())
        print(df)
        if df.index.names != ["entry", "subentry"]:

            # Rebuild the multi index
            keys = ['event', 'subrun', 'run']
            df = rebuild_dataframe_entry_index(df, keys)

        return df
        


class TriggerPrimitivesWorkspace(TriggerAnalysisWorkspace):
    """Workspace for loading MC production TP files (trigger primitives,
    MC truth, neutrinos, particles, IDEs, rawadcs).
    """

    _log = logging.getLogger('TriggerPrimitivesWorkspace')

    tree_names = [
        'event_summary',
        'mctruths',
        'mcneutrinos',
        'mcparticles',
        'simides',
        'simide_summary',
        'tps',
    ]
    _info_name = 'info'

    # TODO: add arguments to disable truth info loading
    def __init__(self, data_path: str, name:str=None, first_entry: int=None, last_entry: int = None, tps_key : str = None, analyzer_name: str = 'triggerAna', tps_folder: str = 'TriggerPrimitives', extra_info: dict = {}):
        super().__init__()

        self._name = name if name is not None else data_path
        self._tuple_rdr = None

        # Labels and ROOT object paths
        self._analyzer_name = analyzer_name
        self._tps_folder  = tps_folder

        self._rawdigits_tree_name: str = 'rawdigis_tree'

        self._data_path = data_path
        self._first_entry = first_entry
        self._last_entry = last_entry

        # Don't forget to copy!
        self._extra_info = extra_info.copy()

        # Per-dataframe post-load decorators
        self._df_decorators = {
            'tps': self._decorate_tps_dataframe,
        }

        # RawADCs registry
        self.rawdigis_events = []

        self.rawdigits_hists = {}
        self._rawadcs = {}

        # Ancillary information
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
        rootfile = self._tuple_rdr.file

        self._log.debug(rootfile.keys())
        self._log.info("Adding processing info")
        self._info = self._tuple_rdr.get_info(self._info_name)

        standard_trees = [t for t in self.tree_names if t != 'tps']

        for t in standard_trees:
            self._log.info(f"Adding '{t}' data")
            try:
                ttree = self._tuple_rdr.get_tree(t)
            except uproot.KeyInFileError:
                self._log.warning(f"Key '{t}' not found in file.")
                ttree = None
            self._trees[t] = ttree

        # Add trigger primitives
        if self._tps_folder in rootfile[f'{self._analyzer_name}']:

            tp_trees_folder = rootfile[f'{self._analyzer_name}/{self._tps_folder}']
            if tps_key:
                logging.info(f'Loading {tps_key}')
                self._trees['tps'] = tp_trees_folder[tps_key]
                self._tps_tree_name = tps_key
            else:
                match len(tp_trees_folder.keys(cycle=False)):
                    case 0:
                        self._trees['tps'] = None
                    case 1:
                        self._tps_tree_name = f'{tp_trees_folder.keys(cycle=False)[0]}'
                        self._tps_tree_path = f'{self._tps_folder}/{self._tps_tree_name}'
                        logging.info(f'Loading {self._tps_tree_path}')
                        self._trees['tps'] = self._tuple_rdr.get_tree(f"{self._tps_tree_path}")
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


    def _load_dataframe(self, name: str) -> TrgDataFrame:
        """Load dataframe from the selected TTree, applying the event cut for this workspace."""
        tree = self._trees[name]

        # Load the tree and convert it into a trigger dataframe
        df = TrgDataFrame(tree.to_df(entry_start=self._first_entry, entry_stop=self._last_entry))

        # Decorate it with the event_uid if not present
        if 'event_uid' not in df.columns:
            df['event_uid'] = df.run.astype("uint64")*1000000+df.subrun.astype("uint64")*100+df.event.astype("uint64")

        # Decorate the dataframe with information
        df.prod_info = self.info
        df.extra_info = self.extra_info
        return df


    # FIXME:
    def _decorate_tps_dataframe(self):
        """Decorate TPS dataframe with extra columns useful for analysis:
        time_peak, sample_start, sample_peak, bt_is_signal.
        """
        df = self._dataframes['tps']
        df['time_peak']    = df.time_start + df.samples_to_peak * 32
        df['sample_start'] = df.time_start // 32
        df['sample_peak']  = df.sample_start + df.samples_to_peak
        df['bt_is_signal'] = (df.bt_numelectrons > 0).astype(np.int8)


    @property
    def name(self):
        return self._name

    @property
    def info(self):
        return self._info

    @property
    def extra_info(self):
        return self._extra_info


    @property
    def tp_maker_name(self):
        return self._tps_tree_name.replace('_', ':')


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
            self._event_list = self._trees['event_summary'].to_df(branches=['event', 'run', 'subrun'], entry_start=self._first_entry, entry_stop=self._last_entry)
        return self._event_list


    #----------------------------------------------------------------------------------------------------------
    # Support for raw dataforms loading
    #
    def add_rawdigits(self, data_path: str):
        """Add a rawdigits (rawadcs) file to the workspace."""

        self._raw_tuple_rdr = RawWaveformsNtupleReader(data_path)

        self._log.info("Loading rawADC tree")
        self._trees['rawdigits'] = self._raw_tuple_rdr.get_tree(self._rawdigits_tree_name)

        self.rawdigits_hists = {}
        for k in self._raw_tuple_rdr.keys(cycle=False):
            obj_name = k.split('/')[-1]
            self._log.info("Retrieving rawADC histograms")

            if obj_name.startswith('ADCsPlane') or obj_name.startswith('ADCsNoisePlane'):
                self.rawdigits_hists[obj_name.split(';')[0]] = self._raw_tuple_rdr[k.split(';')[0]]

        # TODO:
        self._log.info("Load rawdigis event list")
        self.rawdigis_events = self._trees['rawdigits'].event_list()
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

            rwdf = TrgDataFrame(self._trees['rawdigits'].to_df(*ev_uid, channel_mask=channel_mask))
            rwdf.prod_info = {
                'channel_mask': channel_mask
            }
            self._rawadcs[ev_uid] = rwdf

            return self._rawadcs[ev_uid]
