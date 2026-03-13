from pathlib import Path
import tpvalidator

import logging
import tpvalidator.workspace as workspace
from tpvalidator.utils import temporary_log_level
from rich import print
import json

_miniprod_dir = Path(Path(tpvalidator.__file__).parents[2]) / 'data' / 'vd' / 'mini_prod'


radbkg_tawin_dist_file = _miniprod_dir / 'dist' / '1x8x6_radbkg_tawin_dist.root'

tp_presel_spec = _miniprod_dir / 'tp_presel.json'

def load_tp_presel_datasets( selection=None ):

    with open(tp_presel_spec) as json_data:
        cfg = json.load(json_data)

    dataset_path = cfg['dataset_path']
    dataset_info = cfg['dataset_info']
    datasets_spec = cfg['datasets_spec']
    tp_cut = cfg['tp_cut']
    tp_info_update = cfg['tp_info_update']

    if selection:
        dataset_sel = set(selection)
        not_found = dataset_sel.difference(datasets_spec)
        if not_found:
            raise RuntimeError(f'Datasets {not_found} are unknown')

        datasets_spec = { k:v for k,v in datasets_spec.items() if k in dataset_sel }

    from pathlib import Path
    miniprod_dir = _miniprod_dir/ dataset_path

    datasets = {}
    for s, p in datasets_spec.items():
        with temporary_log_level(workspace.TriggerPrimitivesWorkspace._log, logging.WARN):
            ws = workspace.TriggerPrimitivesWorkspace(miniprod_dir / p, extra_info=dataset_info)
        print(f"Dataset '{s}': {ws.num_entries} events")
        print(ws.info)
        datasets[s] = ws


    for n, df in datasets.items():
        df.tps.query(tp_cut, inplace=True)
        df.tps.extra_info.update(tp_info_update)

    return datasets

