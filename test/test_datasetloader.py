#!/usr/bin/env python

import click
from rich import print

from tpvalidator.datacatalogue import load


@click.command()
@click.argument('dataset_spec_path', type=click.Path(file_okay=False, dir_okay=True, exists=True))
def main(dataset_spec_path):


    datasets = load(dataset_spec_path)
    print(datasets)


    for k,ws in datasets.items():
        print(k, ws.num_entries, ws.rawdigis_events, ws.rawdigits_hists)

if __name__ == '__main__':
    main()

