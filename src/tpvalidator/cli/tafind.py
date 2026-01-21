from rich import print
import tpvalidator.tafinder
import logging
import click

import tpvalidator.mcprod.workspace as workspace
from tpvalidator.utilities import temporary_log_level

import tpvalidator.datasets.miniprod as miniprod; 
import pandas as pd
import tpvalidator.tafinder.tpprocessor as tpprocessor

import numpy as np

def test_writing(df: pd.DataFrame):
    import timeit
    
    start_time = timeit.default_timer()
    df.to_feather('xxx.feather')
    end_time = timeit.default_timer()
    print(f"Dataset write time to feather: {end_time - start_time}")

    start_time = timeit.default_timer()
    df = pd.read_feather('xxx.feather')
    end_time = timeit.default_timer()
    print(f"Dataset Load time from feather: {end_time - start_time}")

    start_time = timeit.default_timer()
    df_to_root_typed(df, 'zzz.root', treename='tps')
    end_time = timeit.default_timer()
    print(f"saved TTree file : {end_time - start_time}")


@click.command()
@click.argument('dataset-id')
def main(dataset_id) -> int:

    dn=[dataset_id]
    # dn=['radbkg']


    datasets = miniprod.load_mc_datasets(dn)

    print(f"Loaded {len(datasets)} datasets")

    ws = datasets[dn[0]]

    em_tps = ws.tps

    # test_writing(em_tps)

    print("Processing tps")
    df_writer = tpprocessor.RootDFWriter(f'{dataset_id}.root', 'taFinder')
    # df_writer = None

    swtaf = tpprocessor.SwiftTAFinder(df_writer=df_writer, cfg={'ta_win_sadc_add_bkg': dataset_id != 'radbkg'})


    swtaf.process(em_tps)

    df_writer.write(ws.mctruths, 'mctruths')
    df_writer.write(ws.event_summary, 'event_summary')



    return 0

if __name__ == "__main__":
    raise SystemExit(main())