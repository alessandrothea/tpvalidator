#!/usr/bin/env python

from rich import print
import tpvalidator.workspace as workspace
from tpvalidator.datasetloader import load
from tpvalidator.utils import temporary_log_level


ds = load('./data/vd/soa_trees')
print("tps", ds['gammas'].tps)
print("prod_info", ds['gammas'].tps.prod_info)
print("extra_info", ds['gammas'].tps.extra_info)


print("mctruths", ds['gammas'].mctruths)
print("mctruth_blocks_map", ds['gammas'].mctruth_blocks_map)


print("mctruths", ds['radbkg'].mctruths)
print("mctruth_blocks_map", ds['radbkg'].mctruth_blocks_map)

# with temporary_log_level(workspace.TriggerPrimitivesWorkspace._log, logging.WARNING):
#     ds = load('./data/vd/mini_prod')


