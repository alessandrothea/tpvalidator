from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from rich import print
import click

import tpvalidator.datacatalogue as dctl
import tpvalidator.workspace as workspace
import tpvalidator.algo.tafinder.tpprocessor as tpprocessor


def _iter_batch_params(datasets_dir, dataset_id, batch_size):
    """Yield (trg_file_path, batch_idx, first_entry, last_entry) for each batch."""

    # TODO: find a way to do this better
    ws = dctl.load_datasets(datasets_dir, selection=[dataset_id])[dataset_id]
    first_entry = ws._first_entry
    last_entry = ws._last_entry
    trg_file = ws._data_path
    ws_info = ws.info

    total = ws.num_entries
    # Dismiss the workspace
    del ws
    print(f"Found {total} entries")

    first = first_entry if first_entry is not None else 0
    last = last_entry if last_entry is not None else total

    for batch_idx, i in enumerate(range(first, last, batch_size)):
        yield trg_file, ws_info, batch_idx, i, (i + batch_size if i + batch_size < last else last)


def _process_batch(trg_file, ws_info, batch_idx, first_entry, last_entry, outdir, dataset_id, taf_cfg):
    output_path = f'{outdir}/{dataset_id}_{batch_idx:04d}.root'
    df_writer = tpprocessor.RootDFWriter(output_path, 'taFinder')
    swtaf = tpprocessor.SwiftTAFinder(df_writer=df_writer, cfg=taf_cfg)

    ws = workspace.TriggerPrimitivesWorkspace(trg_file, first_entry=first_entry, last_entry=last_entry, dataset_info=ws_info)
    swtaf.process(ws.tps)

    if ws.mctruths_tree:
        df_writer.write(ws.mctruths, 'mctruths')
    df_writer.write(ws.event_summary, 'event_summary')
    df_writer.write(ws.simide_summary, 'simide_summary')

    from copy import deepcopy
    meta = deepcopy(ws.info)
    
    full_taf_cfg = deepcopy(tpprocessor.default_cfg)
    full_taf_cfg.update(taf_cfg)
    meta['tafinder_cfg'] = full_taf_cfg

    df_writer.writemeta('info', meta)

    print(f"Batch {batch_idx:04d} done → {output_path}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument('datasets-dir', type=str)
@click.argument('dataset-id', type=str)
@click.option('-b', '--batch-size', type=int, default=1000, help='Size of processing batches')
@click.option('-o', '--outdir', default='.', type=click.Path(exists=True, file_okay=False), help='Output folder')
@click.option('-n', '--num-workers', type=int, default=1, help='Number of worker processes')
def main(datasets_dir, dataset_id, batch_size, outdir, num_workers) -> int:

    print(f"Processing tps  [workers={num_workers}]")

    taf_cfg = {
        'ta_inspect_sadc_min': 8000,
        'ta_win_sadc_dist_file': 'ta-win-sadc-vd-1x8x14-radbkg.root',
        'ta_win_sadc_add_bkg': True,
        'ta_inspect_cluster_sadc_threshold': 20750,
        'ta_dbscan_min_neigh': 2
    }

    batch_params = list(_iter_batch_params(datasets_dir, dataset_id, batch_size))
    print(batch_params)


    if len(batch_params) > 1:
        print('Entering in processpool mode')
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_batch, trg_file, ws_info, batch_idx, first_entry, last_entry,
                                outdir, dataset_id, taf_cfg): batch_idx
                for trg_file, ws_info, batch_idx, first_entry, last_entry in batch_params
            }
            for future in as_completed(futures):
                future.result()
    else:
        trg_file, ws_info, batch_idx, first_entry, last_entry = batch_params[0]
        print(">>>>", trg_file, ws_info, batch_idx, first_entry, last_entry)
        _process_batch(trg_file, ws_info, batch_idx, first_entry, last_entry, outdir, dataset_id, taf_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
