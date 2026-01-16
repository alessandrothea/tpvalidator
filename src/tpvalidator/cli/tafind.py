from rich import print
import tpvalidator.tafinder
import logging

import tpvalidator.mcprod.workspace as workspace
from tpvalidator.utilities import temporary_log_level

import tpvalidator.datasets.miniprod as miniprod; 
import pandas as pd

def make_wins(tps: pd.DataFrame, ro_view: int):
    summary = (
        tps
        .query(f'readout_view == {ro_view}')
        .groupby(['event_uid', 'TPCSetID', 'tawin_id'], sort=False)
        .agg(
            n_entries=('tawin_id', "size"),
            sadc=("adc_integral", "sum"),
        )
        .reset_index()
    )

    return summary

def main() -> int:

    datasets = miniprod.load_mc_datasets()
    
    print(f"Loaded {len(datasets)} datasets")


    em_tps = datasets['e-minus'].tps
    rad_tps = datasets['radiols'].tps


    win_size = 1000 # samples

    em_tps['tawin_id'] = (em_tps.sample_peak - 100) // win_size
    rad_tps['tawin_id'] = (rad_tps.sample_peak - 100) // win_size

    em_wins = make_wins(em_tps, 2)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())