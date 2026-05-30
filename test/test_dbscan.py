import numpy as np
import pandas as pd
import pytest

from tpvalidator.algo.tafinder.dbscan import ApplyDBScan


def _make_cluster_df(channel_offsets, sample_offsets, adc_value):
    """Build a small TP DataFrame with tight clusters at given (channel, sample) offsets."""
    rows = []
    for ch_off in channel_offsets:
        for s_off in sample_offsets:
            rows.append({"channel": ch_off, "sample_start": s_off, "adc_integral": adc_value})
    return pd.DataFrame(rows)


class TestApplyDBScan:
    def test_two_separate_clusters_detected(self):
        # Cluster A at channels 0-1, samples 0-3  (tight, << epsilon apart in cm)
        # Cluster B at channels 200-201, samples 200-203  (>> epsilon away)
        df_a = _make_cluster_df([0, 1], [0, 1, 2, 3], adc_value=100)
        df_b = _make_cluster_df([200, 201], [200, 201, 202, 203], adc_value=200)
        df = pd.concat([df_a, df_b], ignore_index=True)

        n_clusters, _, _, _ = ApplyDBScan(df, epsilon=1.5, min_samples=2)
        assert n_clusters == 2

    def test_single_isolated_point_is_noise(self):
        df = pd.DataFrame({"channel": [0], "sample_start": [0], "adc_integral": [50]})
        n_clusters, mean_sadc, total_sadc, max_sadc = ApplyDBScan(df, epsilon=1.5, min_samples=2)
        assert n_clusters == 0
        assert mean_sadc == 0
        assert total_sadc == 0
        assert max_sadc == 0

    def test_cluster_energy_stats(self):
        # Cluster A: 4 points, adc_integral=100 each → cluster sum = 400
        # Cluster B: 4 points, adc_integral=200 each → cluster sum = 800
        df_a = _make_cluster_df([0, 1], [0, 1], adc_value=100)
        df_b = _make_cluster_df([200, 201], [200, 201], adc_value=200)
        df = pd.concat([df_a, df_b], ignore_index=True)

        n_clusters, mean_sadc, total_sadc, max_sadc = ApplyDBScan(df, epsilon=1.5, min_samples=2)
        assert n_clusters == 2
        assert total_sadc == pytest.approx(400 + 800)
        assert max_sadc == pytest.approx(800)
        assert mean_sadc == pytest.approx((400 + 800) / 2)

    def test_one_dense_cluster(self):
        # All points within epsilon of each other
        df = _make_cluster_df([0, 0, 1, 1], [0, 1, 0, 1], adc_value=50)
        n_clusters, mean_sadc, total_sadc, max_sadc = ApplyDBScan(df, epsilon=1.5, min_samples=2)
        assert n_clusters == 1
        assert total_sadc == pytest.approx(50 * len(df))

    def test_does_not_modify_input(self):
        df = pd.DataFrame({"channel": [0, 1], "sample_start": [0, 1], "adc_integral": [10, 20]})
        original_cols = set(df.columns)
        ApplyDBScan(df)
        assert set(df.columns) == original_cols
