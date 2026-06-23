from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from rich import print
import click

import tpvalidator.datacatalogue as dctl
import tpvalidator.workspace as workspace
import tpvalidator.algo.tafinder.tpprocessor as tpprocessor


def _iter_batch_params(datasets_dir, dataset_id, batch_size):
    """Yield (trg_file_path, batch_idx, first_entry, last_entry) for each batch."""
    d = Path(datasets_dir)
    cfg = dctl.parse(datasets_dir)
    dataset_path = d / cfg.dataset_path

    if dataset_id not in cfg.datasets_spec:
        raise KeyError(f'Dataset {dataset_id} not found in {datasets_dir}')

    entry = cfg.datasets_spec[dataset_id]
    trg_file = dataset_path / entry.trg_file

    ws = workspace.TriggerPrimitivesWorkspace(trg_file)
    total = ws.num_entries
    del ws
    print(f"Found {total} entries")

    first = entry.first_entry if entry.first_entry is not None else 0
    last = entry.last_entry if entry.last_entry is not None else total

    for batch_idx, i in enumerate(range(first, last, batch_size)):
        yield trg_file, batch_idx, i, i + batch_size


def _process_batch(trg_file, batch_idx, first_entry, last_entry, outdir, dataset_id, taf_cfg):
    output_path = f'{outdir}/{dataset_id}_{batch_idx:04d}.root'
    df_writer = tpprocessor.RootDFWriter(output_path, 'taFinder')
    swtaf = tpprocessor.SwiftTAFinder(df_writer=df_writer, cfg=taf_cfg)

    ws = workspace.TriggerPrimitivesWorkspace(trg_file, first_entry=first_entry, last_entry=last_entry)
    swtaf.process(ws.tps)

    if ws.mctruths_tree:
        df_writer.write(ws.mctruths, 'mctruths')
    df_writer.write(ws.event_summary, 'event_summary')
    df_writer.writemeta('info', ws.info)

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
    }

    batch_params = list(_iter_batch_params(datasets_dir, dataset_id, batch_size))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_batch, trg_file, batch_idx, first_entry, last_entry,
                            outdir, dataset_id, taf_cfg): batch_idx
            for trg_file, batch_idx, first_entry, last_entry in batch_params
        }
        for future in as_completed(futures):
            future.result()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
