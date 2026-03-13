import logging
import uproot
import awkward as ak
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Union, Dict

_log = logging.getLogger(__name__)

def _check_file_path(file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
       

class NtupleReader:
    """Base class providing ROOT file I/O and path resolution for ntuple readers.

    Opens a ROOT file on construction, exposes it via the :attr:`file`
    property, and delegates unknown attribute access to the underlying uproot
    file object (so ``reader.keys()``, etc. work directly).

    Subclasses add domain-specific methods (e.g. :meth:`get_info` for trigger
    ntuples) and override :meth:`get_tree` to return the appropriate tree
    wrapper.

    Args:
        file_path (str): path to the ROOT file.
        analyzer_dir (str): top-level directory inside the ROOT file used as
            prefix when resolving tree and object paths.

    Attributes:
        file_path (str): path of the opened ROOT file.
        analyzer_dir (str): directory prefix used when resolving object paths.
    """

    def __init__(self, file_path: str, analyzer_dir: str):
        path = Path(file_path)
        _check_file_path(path)

        self.file_path = file_path
        self.analyzer_dir = analyzer_dir
        self._root_file = None

        self._open_file()

    def __getattr__(self, name):
        return getattr(self._root_file, name)
    
    def __getitem__(self, key):
        return self._root_file.__getitem__(key)

    def _open_file(self) -> None:
        try:
            self._root_file = uproot.open(self.file_path)
        except Exception as e:
            _log.error(f"Error loading data from {self.file_path}: {e}")

    @property
    def file(self):
        return self._root_file

    def _get_tree_obj(self, tree_name: str):
        """Resolve ``analyzer_dir/tree_name``, validate it is a TTree, and return it.

        Args:
            tree_name (str): name of the tree inside ``analyzer_dir``.

        Returns:
            Raw uproot TTree object.

        Raises:
            TypeError: if the resolved object is not a TTree.
        """
        tree_path = f"{self.analyzer_dir}/{tree_name}"
        obj = self._root_file[tree_path]
        if not isinstance(obj, uproot.behaviors.TTree.TTree):
            raise TypeError(f"{tree_path} is not a TTree")
        _log.info(f"{tree_path} found with {obj.num_entries} entries")
        return obj



class TriggerTree:
    """Thin wrapper around an uproot TTree that adds a :meth:`to_df` convenience method.

    Unknown attribute access is delegated to the underlying uproot tree, so
    all standard uproot introspection (``keys()``, ``num_entries``, etc.) works
    directly on this object.

    Args:
        tree: open uproot TTree object.
    """

    def __init__(self, tree):
        self._tree = tree

    def __getattr__(self, name):
        # Delegate all unknown attributes to the wrapped tree
        return getattr(self._tree, name)

    def to_df(self, branches=None, entry_start=None, entry_stop=None):
        """Read branches into a pandas DataFrame via awkward-array.

        Args:
            branches (list, optional): branch names to load. Defaults to
                ``None`` (all branches).
            entry_start (int, optional): first entry index to read. Defaults
                to ``None`` (beginning of tree).
            entry_stop (int, optional): one-past-last entry index to read.
                Defaults to ``None`` (end of tree).

        Returns:
            pd.DataFrame with one row per entry (or per element for
            variable-length branches after awkward flattening).
        """
        arr = self._tree.arrays(
            branches,
            entry_start=entry_start,
            entry_stop=entry_stop,
            library="ak"
        )
        return ak.to_dataframe(arr)
    
    def to_df_np(self, branches=None, entry_start=None, entry_stop=None):
        """Read branches into a pandas DataFrame via numumpy-array.

        Args:
            branches (list, optional): branch names to load. Defaults to
                ``None`` (all branches).
            entry_start (int, optional): first entry index to read. Defaults
                to ``None`` (beginning of tree).
            entry_stop (int, optional): one-past-last entry index to read.
                Defaults to ``None`` (end of tree).

        Returns:
            pd.DataFrame with one row per entry (or per element for
            variable-length branches after awkward flattening).
        """
        arr = self._tree.arrays(branches, library="np", entry_start=entry_start, entry_stop=entry_stop, cut=cut)
        df = pd.DataFrame(arr)
        return df.explode(list(df.select_dtypes(include='object').columns))


class TriggerNtupleReader(NtupleReader):
    """Reader for ROOT ntuples produced by the trigger analyser module.

    Exposes trigger-analysis TTrees as :class:`TriggerTree` objects and
    provides access to the processing-info metadata object.  Unknown attribute
    access is delegated to the underlying uproot file object, so standard
    uproot introspection (e.g. ``reader.keys()``) works directly.

    Args:
        file_path (str): path to the ROOT file.
        analyzer_dir (str, optional): top-level directory inside the ROOT file
            that contains the trees and info objects. Defaults to
            ``'triggerAna'``.

    Example::

        reader = TriggerNtupleReader("output.root")
        info   = reader.get_info()
        tree   = reader.get_tree("tptree")
        df     = tree.to_df(branches=["event", "run", "subrun"])
    """

    def __init__(self, file_path: str, analyzer_dir: str = 'triggerAna'):
        super().__init__(file_path, analyzer_dir)

    def get_info(self, info_id: str = 'info') -> Dict:
        """Load processing information from the TNamed info object as a dict.

        Args:
            info_id (str, optional): name of the TNamed object inside
                ``analyzer_dir``. Defaults to ``'info'``.

        Returns:
            dict parsed from the JSON payload stored in the TNamed title.
        """
        info_obj = self._root_file[f"{self.analyzer_dir}/{info_id}"]
        return json.loads(info_obj.members['fTitle'])

    def get_tree(self, tree_name: str) -> TriggerTree:
        """Open a TTree from the analyzer directory and return it as a :class:`TriggerTree`.

        Args:
            tree_name (str): name of the tree inside ``analyzer_dir``
                (e.g. ``'tptree'`` resolves to ``'triggerAna/tptree'``).

        Returns:
            :class:`TriggerTree` wrapping the requested uproot TTree.

        Raises:
            TypeError: if the object at the resolved path is not a TTree.
        """
        return TriggerTree(self._get_tree_obj(tree_name))

    def read_tree_old(self, tree_name: str, branch_names: Optional[list] = None, entry_start: Optional[int] = None, entry_stop: Optional[int] = None, cut: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load an uproot TTree into a DataFrame.

        Args:
            tree_name: name or path of the tree in the ROOT file
            trg_dir (str, optional): directory containing the tree. Defaults to 'triggerAna'.
            cut (str, optional): uproot cut expression to filter rows. Defaults to None.
            branch_names (list, optional): branches to load. Defaults to None (all branches).

        Returns:
            pd.DataFrame or None on error.
        """
        try:
            tree_path = f"{self.analyzer_dir}/{tree_name}"
            tree = self._root_file[tree_path]
            _log.info(f"{tree_path} found with {tree.num_entries} ")
            arr = tree.arrays(branch_names, library="ak", entry_start=entry_start, entry_stop=entry_stop, cut=cut)
            return ak.to_dataframe(arr)
        except Exception as e:
            _log.error(f"Error loading tree {tree_name} from {self.file_path}: {e}")
            return None

    def read_tree_np(self, tree_name: str, branch_names: Optional[list] = None, entry_start: Optional[int] = None, entry_stop: Optional[int] = None, cut: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load an uproot TTree into a DataFrame.

        Args:
            tree_name: name or path of the tree in the ROOT file
            trg_dir (str, optional): directory containing the tree. Defaults to 'triggerAna'.
            cut (str, optional): uproot cut expression to filter rows. Defaults to None.
            branch_names (list, optional): branches to load. Defaults to None (all branches).

        Returns:
            pd.DataFrame or None on error.
        """
        try:
            tree = self._root_file[f"{self.analyzer_dir}/{tree_name}"]
            arr = tree.arrays(branch_names, library="np", entry_start=entry_start, entry_stop=entry_stop, cut=cut)
            df = pd.DataFrame(arr)
            df = df.explode(list(df.select_dtypes(include='object').columns))

            return df
        except Exception as e:
            _log.error(f"Error loading tree {tree_name} from {self.file_path}: {e}")
            return None

#-----


class RawWaveformsTree:
    """Wrapper around an uproot TTree holding raw waveform (rawdigits) data.

    Handles both dense trees (every channel stored as a branch) and sparse
    trees (only active channels stored, with an extra branch listing them).
    The correct loading strategy is detected automatically via
    :meth:`_find_active_channels_branch`.

    Unknown attribute access is delegated to the underlying uproot tree, so
    all standard uproot introspection (``keys()``, ``num_entries``, etc.) works
    directly on this object.

    Args:
        tree: open uproot TTree object for raw waveform data.
    """

    def __init__(self, tree):
        self._tree = tree

    def __getattr__(self, name):
        # Delegate all unknown attributes to the wrapped tree
        return getattr(self._tree, name)

    def event_list(self):    
        return self._tree.arrays(['event', 'run', 'subrun'], library='pd').drop_duplicates()


    def to_df(self, ev: int):
        """Load waveform data for a given event into a pandas DataFrame.

        Dispatches to sparse or dense loading based on whether an
        active-channels branch is present in the tree.

        Args:
            ev (int): event index to load (currently only ``1``, the first
                event, is supported).

        Returns:
            pd.DataFrame with columns ``[event, run, subrun, <channel_ids>,
            sample_id]``, one row per ADC sample.
        """
        if (self._find_active_channels_branch()):
            df_wf = self._load_sparse_waveform_data(ev)
        else:
            df_wf = self._load_waveform_data(ev)

        return df_wf
    
    def _find_active_channels_branch(self) -> Optional[str]:
        """Return the name of the active-channels branch in a sparse waveform tree, or None."""
        for name in ['active_channels', 'chans_with_electrons']:
            if name in self._tree.keys():
                return name
        return None
    
    def _load_waveform_data(self, ev_sel: Union[int, list] = 1):
        """Load dense waveform data for all channels into a pandas DataFrame.

        Reads every branch that is not ``event``, ``run``, or ``subrun`` as a
        channel waveform.  All channel arrays are exploded into one row per ADC
        sample and cast to ``uint16``.  A ``sample_id`` column (0-based) is
        appended.

        Args:
            ev_sel: event selection. Only the first event (``1``) is currently
                supported; passing any other value raises ``RuntimeError``.

        Returns:
            pd.DataFrame with columns ``[event, run, subrun, <channel_ids>,
            sample_id]``, one row per ADC sample.
        """

        if not (type(ev_sel) == int and ev_sel == 1):
            raise RuntimeError("Only the loading of the first event is supported")

        chans = [o.name for o in self._tree.branches if o.name not in ['event', 'run', 'subrun']]
        _log.debug(f"found {len(chans)} channels")

        _log.debug("Loading tree into np arrays")
        arrays = self._tree.arrays(library='np')
        _log.debug("Done loading tree into np arrays")

        _log.debug("Converting np arrays to dataframe")
        df = pd.DataFrame(arrays)
        _log.debug("Done converting np arrays to dataframe")

        df.columns = [int(c) if c not in ["event", "run", "subrun"] else c for c in df.columns]

        _log.debug("Expanding waveforms")
        df_waveforms = df.explode(chans)
        _log.debug("Done expanding waveforms")

        df_waveforms = df_waveforms.astype({c: 'uint16' for c in chans})
        df_waveforms['sample_id'] = np.arange(0, len(df_waveforms))

        return df_waveforms


    def _load_sparse_waveform_data(self, ev_sel: Union[int, list] = 1) -> Optional[pd.DataFrame]:
        """Load sparse waveform data, reading only channels listed in the active-channels branch.

        Reads the active-channels branch to determine which channel branches to
        load, then explodes them into one row per ADC sample and casts to
        ``uint16``.  A ``sample_id`` column (0-based) is appended.  Returns
        ``None`` on error.

        Args:
            ev_sel: event selection. Only the first event (``1``) is currently
                supported; passing any other value raises ``RuntimeError``.

        Returns:
            pd.DataFrame with columns ``[event, run, subrun, <channel_ids>,
            sample_id]``, one row per ADC sample, or ``None`` on error.
        """
        try:
            activ_chans_branch = self._find_active_channels_branch()
            if activ_chans_branch is None:
                raise RuntimeError(
                    "Active channel branch not found in tree. "
                    "This doesn't look like a sparse waveform tree."
                )

            branches = ["event", "run", "subrun", activ_chans_branch]
            df_evs = self._tree.arrays(branches, library='np')
            df_evs = pd.DataFrame(df_evs)

            if not (type(ev_sel) == int and ev_sel == 1):
                raise RuntimeError("Only the loading of the first event is supported")

            ev_num = df_evs.event[0]
            chans = list(df_evs[df_evs.event == ev_num][activ_chans_branch][0])
            _log.debug(f"found {len(chans)} channels")

            _log.debug("Loading tree into np arrays")
            arrays = self._tree.arrays(["event", "run", "subrun"] + [str(c) for c in chans], library='np')
            _log.debug("Done loading tree into np arrays")

            # Convert channel column names to integer
            for c in chans:
                arrays[int(c)] = arrays.pop(str(c))

            _log.debug("Converting np arrays to dataframe")
            # Build dataframe
            df = pd.DataFrame(arrays)
            _log.debug("Done converting np arrays to dataframe")

            # Flatten dataframe structure
            _log.debug("Expanding waveforms")
            df = df.explode([int(c) for c in chans])
            _log.debug("Done expanding waveforms")

            # Add the sample id
            df['sample_id'] = np.arange(0, len(df))

            # Change channel column types to unsigned ints
            obj_cols = df.select_dtypes(include='object').columns
            df[obj_cols] = df[obj_cols].astype('uint16')
           
            return df

        except Exception as e:
            _log.error(f"Error loading sparse waveform data: {e}")
            return None


    

class RawWaveformsNtupleReader(NtupleReader):
    """Reader for ROOT files containing raw waveform (rawdigits) trees.

    Exposes waveform TTrees as :class:`RawWaveformsTree` objects, which handle
    both dense and sparse channel layouts.  Only :meth:`get_tree` is provided;
    trigger-analysis methods (``get_info``, etc.) are intentionally absent.

    Args:
        file_path (str): path to the ROOT file.
        analyzer_dir (str, optional): top-level directory inside the ROOT file.
            Defaults to ``'triggerana'``.
    """

    def __init__(self, file_path: str, analyzer_dir: str = 'triggerana'):
        super().__init__(file_path, analyzer_dir)

    def get_tree(self, tree_name: str) -> RawWaveformsTree:
        """Open a TTree from the analyzer directory and return it as a :class:`RawWaveformsTree`.

        Args:
            tree_name (str): name of the tree inside ``analyzer_dir``
                (e.g. ``'rawdigis_tree'`` resolves to ``'triggerana/rawdigis_tree'``).

        Returns:
            :class:`RawWaveformsTree` wrapping the requested uproot TTree.

        Raises:
            TypeError: if the object at the resolved path is not a TTree.
        """
        return RawWaveformsTree(self._get_tree_obj(tree_name))

#-----



def find_active_channels_branch(tree) -> Optional[str]:
    """Return the name of the active-channels branch in a sparse waveform tree, or None."""
    for name in ['active_channels', 'chans_with_electrons']:
        if name in tree.keys():
            return name
    return None


# def read_data(file_path: str, tree_name: str = 'triggerana/tree', branch_names: Optional[list] = None, max_events=None) -> Optional[pd.DataFrame]:
#     """Load data from a ROOT tree into a DataFrame after expanding vectors into rows.

#     Args:
#         file_path (str): path to the ROOT file containing the ROOT tree
#         tree_name (str): name or path of the tree in the ROOT file
#         branch_names (list, optional): branches to import. Defaults to None (all).
#         max_events (int, optional): maximum number of events. Defaults to None.

#     Returns:
#         pd.DataFrame or None on error.
#     """
#     _check_file_path(Path(file_path))

#     try:
#         with uproot.open(f'{file_path}:{tree_name}') as tree:
#             arr = tree.arrays(branch_names, library="ak", entry_stop=max_events)
#             return ak.to_dataframe(arr)
#     except Exception as e:
#         _log.error(f"Error loading data from {file_path}: {e}")
#         return None



# def read_event_list(file_path: str, tree_name: str) -> Optional[pd.DataFrame]:
#     """Load the event/run/subrun index from a ROOT tree.

#     Args:
#         file_path (str): path to the ROOT file
#         tree_name (str): name or path of the tree

#     Returns:
#         pd.DataFrame with columns [event, run, subrun], or None on error.
#     """
#     _check_file_path(Path(file_path))

#     try:
#         with uproot.open(f'{file_path}:{tree_name}') as tree:
#             return tree.arrays(["event", "run", "subrun"], library='pd')
#     except Exception as e:
#         _log.error(f"Error loading event list from {file_path}: {e}")
#         return None


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
    _check_file_path(Path(file_path))

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


# def read_waveforms(filepath: str, channel_ids: list, tree_name: str = 'triggerana/rawdigis_tree', max_events: int = 1, first_event: int = 0) -> Optional[pd.DataFrame]:
#     """Load waveform data for specified channels from a ROOT file.

#     Args:
#         filepath (str): path to the ROOT file containing waveform data.
#         channel_ids (list): list of channel IDs to load waveforms for.
#         tree_name (str, optional): name of the tree. Defaults to 'triggerana/rawdigis_tree'.
#         max_events (int, optional): maximum number of events to load. Defaults to 1.
#         first_event (int, optional): index of the first event to load. Defaults to 0.

#     Returns:
#         pd.DataFrame or None on error.
#     """
#     try:
#         branch_names = [f"{ch:d}" for ch in channel_ids]
#         with uproot.open(f'{filepath}:{tree_name}') as tree:
#             arrays = tree.arrays(branch_names, library="ak", entry_stop=max_events)
#             df = ak.to_dataframe(arrays)
#             df.columns = [int(c) for c in df.columns]
#             df.index = np.arange(0, len(df))
#             return df
#     except Exception as e:
#         _log.error(f"Error loading waveform data from {filepath}: {e}")
#         return None
