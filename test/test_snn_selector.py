"""Tests for TPSignalNoiseSelector (no visualization methods tested)."""
import pytest

from tpvalidator.analysis.snn import TPSignalNoiseSelector


class TestTPSignalNoiseSelector:
    # synthetic_tps fixture from conftest.py:
    #   300 rows, readout_view cycles [0,1,2], bt_is_signal: first 150 = 0, last 150 = 1
    #   → each view: 100 rows, 50 noise, 50 signal

    def test_len_equals_total_rows(self, synthetic_tps):
        sel = TPSignalNoiseSelector(synthetic_tps)
        print(f"\nlen(sel): {len(sel)}, len(synthetic_tps): {len(synthetic_tps)}")
        assert len(sel) == len(synthetic_tps)

    def test_all_contains_all_rows(self, synthetic_tps):
        sel = TPSignalNoiseSelector(synthetic_tps)
        print(f"\nlen(sel.all): {len(sel.all)}, expected: {len(synthetic_tps)}")
        assert len(sel.all) == len(synthetic_tps)

    def test_view_split_covers_all_rows(self, synthetic_tps):
        sel = TPSignalNoiseSelector(synthetic_tps)
        total = len(sel.all_view_0) + len(sel.all_view_1) + len(sel.all_view_2)
        print(f"\nview0={len(sel.all_view_0)}, view1={len(sel.all_view_1)}, "
              f"view2={len(sel.all_view_2)}, total={total}, expected={len(synthetic_tps)}")
        assert total == len(synthetic_tps)

    def test_each_view_has_correct_count(self, synthetic_tps):
        sel = TPSignalNoiseSelector(synthetic_tps)
        print(f"\nview0={len(sel.all_view_0)}, view1={len(sel.all_view_1)}, "
              f"view2={len(sel.all_view_2)}, all expected 100")
        assert len(sel.all_view_0) == 100
        assert len(sel.all_view_1) == 100
        assert len(sel.all_view_2) == 100

    def test_signal_noise_partition_per_view(self, synthetic_tps):
        sel = TPSignalNoiseSelector(synthetic_tps)
        for v in (0, 1, 2):
            sig = sel.sig_by_view[v]
            noi = sel.noise_by_view[v]
            all_v = sel.all_by_view[v]
            print(f"\nview{v}: sig={len(sig)}, noise={len(noi)}, total={len(all_v)}")
            assert len(sig) + len(noi) == len(all_v)

    def test_signal_has_nonzero_bt_is_signal(self, synthetic_tps):
        sel = TPSignalNoiseSelector(synthetic_tps)
        vals = sel.sig_view_0["bt_is_signal"]
        print(f"\nsig_view_0 bt_is_signal unique values: {vals.unique()}")
        assert (vals != 0).all()

    def test_noise_has_zero_bt_is_signal(self, synthetic_tps):
        sel = TPSignalNoiseSelector(synthetic_tps)
        vals = sel.noise_view_0["bt_is_signal"]
        print(f"\nnoise_view_0 bt_is_signal unique values: {vals.unique()}")
        assert (vals == 0).all()

    def test_legacy_aliases_point_to_same_data(self, synthetic_tps):
        sel = TPSignalNoiseSelector(synthetic_tps)
        print(f"\nsel.p0 is sel.all_view_0: {sel.p0 is sel.all_view_0}")
        print(f"sel.sig_p1 is sel.sig_view_1: {sel.sig_p1 is sel.sig_view_1}")
        print(f"sel.noise_p2 is sel.noise_view_2: {sel.noise_p2 is sel.noise_view_2}")
        assert sel.p0 is sel.all_view_0
        assert sel.sig_p1 is sel.sig_view_1
        assert sel.noise_p2 is sel.noise_view_2

    def test_query_reduces_rows(self, synthetic_tps):
        sel = TPSignalNoiseSelector(synthetic_tps)
        filtered = sel.query("bt_is_signal == 1")
        print(f"\nlen(sel): {len(sel)}, len(filtered): {len(filtered)}, "
              f"type: {type(filtered).__name__}")
        assert len(filtered) < len(sel)
        assert isinstance(filtered, TPSignalNoiseSelector)

    def test_query_result_is_selector(self, synthetic_tps):
        sel = TPSignalNoiseSelector(synthetic_tps)
        result = sel.query("readout_view == 0")
        unique_views = result.all["readout_view"].unique()
        print(f"\ntype: {type(result).__name__}, unique readout_view: {unique_views}")
        assert isinstance(result, TPSignalNoiseSelector)
        assert (result.all["readout_view"] == 0).all()
