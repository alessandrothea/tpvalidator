
import logging
import uproot
import json
import pandas as pd
import numpy as np

from rich import print
from typing import Tuple, Optional, Union, Sequence, Dict

class TrgDataFrame2g(pd.DataFrame):
    # normal properties
    _metadata = ["prod_info", 'extra_info']

    @property
    def _constructor(self):
        return TrgDataFrame2g
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prod_info = {}
        self.extra_info = {}


class TriggerAnalysisWorkspace:
    """Workspace class for trigger objects analysis

    Raises:
        AttributeError: _description_

    Returns:
        _type_: _description_
    """

    _log = logging.getLogger('TriggerAnalsysWorkspace')

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
                    # Build tree name
                    # Fetch the tree object
                    ttree = f[ttree_name]

                    self._log.info(f"{ttree_name} found with {ttree.num_entries} rows")
                except uproot.KeyInFileError:
                    self._log.warning(f"Key '{ttree_name}' not found.")
                    ttree = None
                # setattr(self, f'{t}_tree', ttree) 
                self._trees[t] = ttree

    def __getattr__(self, name):
        if name in self.tree_names:
            return self._get_dataframe(name)
        else:
            raise AttributeError(name)


    def _load_dataframe(self, tree):
        cut=None
        df = TrgDataFrame2g(tree.arrays(library="np", cut=cut))
        return df

        
    def _get_dataframe(self, name) -> TrgDataFrame2g:

        df = self._dataframes.get(name, None)
        if df is None:
            df = self._load_dataframe(self._trees[name])
            self._dataframes[name] = df
        
        return df
