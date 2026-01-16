from pathlib import Path
import tpvalidator

import logging
import tpvalidator.mcprod.workspace as workspace
from tpvalidator.utilities import temporary_log_level
from rich import print

_miniprod_dir = Path(Path(tpvalidator.__file__).parents[2]) / 'data' / 'vd' / 'mini_prod'

def load_mc_datasets():

    dataset_info = {
        'readout_window' : 8500
    }

    from pathlib import Path
    miniprod_dir = _miniprod_dir/ 'tp_presel'

    with temporary_log_level(workspace.TriggerPrimitivesWorkspace._log, logging.WARN):
        em_ws = workspace.TriggerPrimitivesWorkspace(miniprod_dir / 'vd_1x8x6_eminus_center_2333289_tppresel_ana.ntuple.root', extra_info=dataset_info)
    print(f"Dataset e-minus: {em_ws.num_events} events")
    print(em_ws.info)


    with temporary_log_level(workspace.TriggerPrimitivesWorkspace._log, logging.WARN):
        gm_ws = workspace.TriggerPrimitivesWorkspace(miniprod_dir / 'vd_1x8x6_gamma_center_2333392_tppresel_ana.ntuple.root', extra_info=dataset_info)
    print(f"Dataset gamma: {gm_ws.num_events} events")
    print(gm_ws.info)


    with temporary_log_level(workspace.TriggerPrimitivesWorkspace._log, logging.WARN):
        mu_ws = workspace.TriggerPrimitivesWorkspace(miniprod_dir / 'vd_1x8x6_muminus_center_2333393_tppresel_ana.ntuple.root', extra_info=dataset_info)
    print(f"Dataset mu-minus: {mu_ws.num_events} events")
    print(mu_ws.info)


    with temporary_log_level(workspace.TriggerPrimitivesWorkspace._log, logging.WARN):
        rad_ws = workspace.TriggerPrimitivesWorkspace(miniprod_dir / 'vd_1x8x6_radbkg_2333394_23335306_233378794_tppresel_ana.ntuple.root', extra_info=dataset_info)
    print(f"Dataset radiols: {rad_ws.num_events} events")
    print(rad_ws.info)

    datasets = {
        'e-minus': em_ws,
        'gamma': gm_ws,
        'mu-minus': mu_ws,
        'radbkg': rad_ws,
    }

    wirecell_ides_cut = 'sample_peak > 100 & sample_peak < 8100'

    for n, df in datasets.items():
        df.tps.query(wirecell_ides_cut, inplace=True)
        df.tps.extra_info['readout_window'] = 8000

    return datasets