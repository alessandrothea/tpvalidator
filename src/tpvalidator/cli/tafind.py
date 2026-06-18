from rich import print
# import tpvalidator.algo.tafinder
# import logging
import click

# import tpvalidator.workspace as workspace
# from tpvalidator.utils import temporary_log_level

# import tpvalidator.datasets.miniprod as miniprod
import tpvalidator.datacatalogue as dctl
import pandas as pd
import tpvalidator.algo.tafinder.tpprocessor as tpprocessor

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
@click.argument('datasets-dir', type=str)
@click.argument('dataset-id', type=str)
@click.option('-o', '--outdir', default='.', type=click.Path(exists=True, file_okay=False))
def main(datasets_dir, dataset_id, outdir) -> int:


    # dn=[dataset_id]
    # dn=['radbkg']


    # datasets = miniprod.load_tp_presel_datasets(dn)
    datasets = dctl.load(datasets_dir)


    print(f"Loaded {len(datasets)} datasets")

    ws = datasets[dataset_id]

    em_tps = ws.tps


    print("Processing tps")
    df_writer = tpprocessor.RootDFWriter(outdir + f'/{dataset_id}.root', 'taFinder')

    taf_cfg = {
        'ta_inspect_sadc_min': 11000,
        # 'ta_win_sadc_add_bkg': dataset_id != 'radbkg',
        # 'ta_win_sadc_dist_file': miniprod.radbkg_tawin_dist_file
    }

    swtaf = tpprocessor.SwiftTAFinder(df_writer=df_writer, cfg=taf_cfg)


    swtaf.process(em_tps)

    if ws.mctruths_tree:
        df_writer.write(ws.mctruths, 'mctruths')
    df_writer.write(ws.event_summary, 'event_summary')



    return 0

if __name__ == "__main__":
    raise SystemExit(main())