from pathlib import Path
from rich import print
import click
import uproot
from tpvalidator.workspace import TriggerActivityWorkspace
from tpvalidator.analysis.histograms import build_histogram, make_regaxis

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument('tawin_file', type=click.Path(exists=True, dir_okay=False))
@click.option('-s', '--sadc-hist-file', type=click.Path(exists=False, dir_okay=False))
def main(tawin_file, sadc_hist_file):
    
    print(f'Loading {sadc_hist_file}')
    ws = TriggerActivityWorkspace(tawin_file)


    # Create the tawindo
    ta_win_stats = ws.ta_win_stats

    ta_sadc_axis = make_regaxis(ta_win_stats, 'sadc', 100)

    
    h_sadc = build_histogram(ws.ta_win_stats, [ta_sadc_axis])

    print(h_sadc)


    if sadc_hist_file:
        with uproot.recreate(sadc_hist_file,) as f:
            f['h_tawin_sadc'] = h_sadc





if __name__ == '__main':
    main()