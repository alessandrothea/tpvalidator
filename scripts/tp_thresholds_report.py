#!/usr/bin/env python
"""Generate a PDF validation report for Ar39 noise studies."""

import json
# import functools
import logging

import matplotlib.pyplot as plt

from fpdf import FontFace, TextStyle, Align
from pathlib import Path
from rich import print
from rich.logging import RichHandler

import click

import tpvalidator.workspace as workspace
import tpvalidator.analysis.snn as snn
from tpvalidator.viz.backtracker import BackTrackerPlotter
from tpvalidator.utils import temporary_log_level
from tpvalidator.report.portfolio import Portfolio
from tpvalidator.report.pdf import ReportPDF, load_report_fonts



@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("dataset_dir_path", type=click.Path(dir_okay=True, file_okay=False, exists=True))
# @click.option('-d', '--dataset-name', 'dataset_name', type=str, default='ar39_5e_00',
#               help="Path to the raw-waveform ROOT file.")
# @click.option('-i', '--interactive', is_flag=True, default=False,
#               help="Drop into an IPython shell after loading data.")
# @click.option('--figs/--no-figs', 'make_figures', default=True, show_default=True,
#               help="Generate figures (disable to re-use a previous run's figures).")
# @click.option('--report/--no-report', 'make_report', default=True, show_default=True,
#               help="Assemble and write the PDF report.")
@click.option('-o', 'output_dir', type=click.Path(file_okay=False), default='./reports/tp_thresholds/',
              show_default=True)
def cli(
    dataset_dir_path, 
    # dataset_name, 
    output_dir,
    # make_figures, 
    # make_report, 
    # interactive
    ):

    report_dir = Path(output_dir)
    figures_dir = report_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    notes_file = figures_dir / 'notes.json'