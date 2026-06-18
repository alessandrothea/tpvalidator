from ..detgeometry import get_by_geocfg_id

class TrgWorkspaceAnalyzer:
    """Base class for analysers

    Returns:
        _type_: _description_
    """

    #-----------
    def __init__(self, ws):
        
        self._ws = ws
 
       # Initialize geometry
        self._geo = get_by_geocfg_id(ws.info['geo']['detector'])

    def simulated_readout_time(self) -> float:
        """Return total simulated time in seconds.

        Computed as ``2 × readout_window × num_entries × 0.5 µs``, where the
        factor of 2 accounts for pre- and post-spill readout windows.
        Always derived from ``ws.mctruths`` regardless of the active
        collection.
        """
        
        sampling_time = 0.5e-6  # Sampling time 1/2 usec
        ro_win = self.ws.extra_info['readout_window']
        num_entries = self.ws.extra_info['num_entries']
        return ro_win * sampling_time * num_entries

    @property
    def ws(self):
        return self._ws

    @property
    def geo(self):
        return self._geo