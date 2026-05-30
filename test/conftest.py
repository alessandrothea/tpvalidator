import numpy as np
import pandas as pd
import hist
import pytest


@pytest.fixture
def uniform_hist_1d():
    h = hist.Hist(hist.axis.Regular(10, 0, 10, name="x"))
    h.fill(x=np.linspace(0.5, 9.5, 100))
    return h


@pytest.fixture
def uniform_hist_2d():
    rng = np.random.default_rng(0)
    h = hist.Hist(
        hist.axis.Regular(10, 0, 10, name="x"),
        hist.axis.Regular(5, 0, 5, name="y"),
    )
    h.fill(x=rng.uniform(0, 10, 1000), y=rng.uniform(0, 5, 1000))
    return h


@pytest.fixture
def synthetic_tps():
    rng = np.random.default_rng(1)
    n = 300
    return pd.DataFrame({
        "readout_view":           np.tile([0, 1, 2], n // 3),
        "bt_is_signal":           np.repeat([0, 1], n // 2),
        "channel":                rng.integers(0, 480, n),
        "sample_peak":            rng.integers(0, 6000, n),
        "adc_peak":               rng.integers(10, 200, n),
        "adc_integral":           rng.integers(50, 1000, n),
        "samples_over_threshold": rng.integers(1, 20, n),
    })
