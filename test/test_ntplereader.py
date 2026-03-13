#!/usr/bin/env python

from rich import print

from tpvalidator.rootio import TriggerNtupleReader
r = TriggerNtupleReader('./data/vd/soa_trees/tps/anatree_vd_gamma_hist.root')
print(r.read_info())
# print(r.read_tree('event_summary', entry_stop=2))
print(r.read_tree('event_summary', entry_stop=1))
# print(r.read_tree('event_summary', entry_start=10, entry_stop=11))



# print(r.read_tree('mcparticles', entry_start=1, entry_stop=2))
print(r.read_tree('TriggerPrimitives/tpmakerTPCSimpleThreshold__TriggerPrimitiveMaker', entry_start=1, entry_stop=2))
print(r.read_tree_b('TriggerPrimitives/tpmakerTPCSimpleThreshold__TriggerPrimitiveMaker', entry_start=1, entry_stop=2))