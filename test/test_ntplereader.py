#!/usr/bin/env python

from rich import print

from tpvalidator.rootio import TriggerNtupleReader, RawWaveformsNtupleReader
r = TriggerNtupleReader('./data/vd/soa_trees/tps/anatree_vd_gamma_hist.root')
print(r.get_info())
# print(r.get_tree('event_summary', entry_stop=2))
print(r.get_tree('event_summary', entry_stop=1))
# print(r.get_tree('event_summary', entry_start=10, entry_stop=11))



# print(r.get_tree('mcparticles', entry_start=1, entry_stop=2))
print(r.get_tree('TriggerPrimitives/tpmakerTPCSimpleThreshold__TriggerPrimitiveMaker', entry_start=1, entry_stop=2))
print(r.get_tree_b('TriggerPrimitives/tpmakerTPCSimpleThreshold__TriggerPrimitiveMaker', entry_start=1, entry_stop=2))




rr = RawWaveformsNtupleReader('../data/vd/ar39/100events/trigger_digits_waves_detsim_vd_ar39.root')