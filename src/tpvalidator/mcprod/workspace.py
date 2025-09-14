
import logging
import uproot

from rich import print

class TriggerPrimitivesWorkspace:
    
    _log = logging.getLogger('TriggerPrimitivesWorkspace')

    _analyzer_name: str = 'triggerAnaDumpAll'
    _mctruth_tree_name: str = f'{_analyzer_name}/mctruths'
    _mcneutrinos_tree_name: str = f'{_analyzer_name}/mcneutrinos'
    _mcparticles_tree_name: str = f'{_analyzer_name}/mcparticles'
    _ide_tree_name: str = f'{_analyzer_name}/simides'


    # TODO: add arguments to disable truth info loading
    def __init__(self, data_path: str, first_entry: int=0, last_entry: int = None):

        self._data_path = data_path
        self._first_entry = first_entry
        self._last_entry = last_entry

        self._log.info("Opening Trigger Primitives file")
        with uproot.open(self._data_path) as f:
            
            #
            # Add Montecarlo tree objets
            #
            self._log.debug(f.keys())
            
            # Try loading the MCTruth tree
            self._log.info("Adding MCTruth data")
            try:
                self.mctruth_tree = f[self._mctruth_tree_name]
                self._log.info(f"{self._mctruth_tree_name} found with {self.mctruth_tree.num_entries} rows")

            except uproot.KeyInFileError:
                self._log.warning(f"Key '{self._mctruth_tree_name}' not found in file.")
                self.mctruth_tree = None

            # Try loading the MCTruth tree
            self._log.info("Adding MCNeutrino data")
            try:
                self.mcneutrino_tree = f[self._mcneutrinos_tree_name]
                self._log.info(f"{self._mcneutrinos_tree_name} found with {self.mcneutrino_tree.num_entries} rows")

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
                self.ides_tree = f[self._ide_tree_name]
                self._log.info(f"{self._ide_tree_name} found with {self.ides_tree.num_entries} rows")

            except uproot.KeyInFileError:
                self._log.warning(f"Key '{self._ide_tree_name}' not found in file.")
                self.ides_tree = None


            # Add trigger primitives


    def _events(self, tree):
        return tree.arrays(['Event'], library='pd').Event.unique()

    def get_event_cut(self, tree):

        ev_list = self._events(tree)

        cuts = []
        if not self._first_entry is None:
            # First entry goes from 0 t N, positive
            cuts.append(f'(Event >= {ev_list[self._first_entry]})')
        
        if not self._last_entry is None:
            cuts.append(f'(Event <= {ev_list[self._last_entry]})')

        return ' & '.join(cuts) if len(cuts) > 0 else None

    def get_mctruth(self):
        if getattr(self, '_mctruth', None) is None:
            self._log.debug("Loading MCTruth dataset")
            self._mctruth = self._load_dataframe('mctruth')

        return self._mctruth

    def get_mcneutrino(self):
        if getattr(self, '_mcneutrino', None) is None:
            self._log.debug("Loading MCNeutrino dataset")
            self._mcneutrino = self._load_dataframe('mcneutrino')
        return self._mcneutrino
    
    def get_mcparticles(self):
        if getattr(self, '_mcparticles', None) is None:
            self._log.debug("Loading MCParticles dataset")
            self._mcparticles = self._load_dataframe('mcparticles')

        return self._mcparticles
    
    def get_ides(self):
        if getattr(self, '_ides', None) is None:
            self._log.debug("Loading IDEs dataset")
            self._ides = self._load_dataframe('ides')
        return self._ides
    

    def _load_dataframe(self, df_id):
        tree = getattr(self, f'{df_id}_tree')
        ev_cut = self.get_event_cut(tree) if tree.num_entries > 0 else None
        self._log.debug(f"Applying event cut for {df_id}")
        return tree.arrays(library="pd", cut=ev_cut)
