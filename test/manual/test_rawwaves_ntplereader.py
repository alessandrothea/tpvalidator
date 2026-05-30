#!/usr/bin/env python

from rich import print

import tpvalidator.rootio as rootio
import logging
from tpvalidator.rootio import TriggerNtupleReader, RawWaveformsNtupleReader
from tpvalidator.utils import temporary_log_level

rr = RawWaveformsNtupleReader('data/vd/ar39/100events/trigger_digits_waves_detsim_vd_ar39.root')

rt = rr.get_tree('rawdigis_tree')

print('num entries', rt.num_entries)
with temporary_log_level(rootio._log, logging.DEBUG):
    df = rt.to_df(1)
print(df)