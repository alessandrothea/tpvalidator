"""
Tests for tpc_angles.py.

Note: calculate_angles / calculate_more_angles use the pandas Series .abs() method
internally, so they require pandas Series (not plain numpy arrays) as inputs.
"""
import numpy as np
import pandas as pd
import pytest

from tpvalidator.tpc_angles import unit_vector, wrap_phi, calculate_angles


class TestUnitVector:
    def test_normalises_to_magnitude_one(self):
        v = np.array([3.0, 4.0, 0.0])
        u = unit_vector(v)
        assert np.linalg.norm(u) == pytest.approx(1.0)

    def test_already_unit(self):
        v = np.array([1.0, 0.0, 0.0])
        assert np.allclose(unit_vector(v), v)

    def test_direction_preserved(self):
        v = np.array([0.0, 0.0, 5.0])
        u = unit_vector(v)
        assert np.allclose(u, [0, 0, 1])


class TestWrapPhi:
    def test_negative_angle(self):
        assert wrap_phi(-45.0) == pytest.approx(45.0)

    def test_positive_angle(self):
        assert wrap_phi(30.0) == pytest.approx(30.0)

    def test_zero(self):
        assert wrap_phi(0.0) == pytest.approx(0.0)


class TestCalculateAngles:
    def _series(self, *vals):
        return tuple(pd.Series([v]) for v in vals)

    def test_forward_particle_xz_angle_is_zero(self):
        # pz=1, px=py=0 → no transverse component → theta_xz == 0
        px, py, pz, p_mag = self._series(0.0, 0.0, 1.0, 1.0)
        _, _, _, theta_xz, _, _ = calculate_angles(px, py, pz, p_mag, "hd")
        assert float(theta_xz.iloc[0]) == pytest.approx(0.0, abs=1e-10)

    def test_transverse_particle_xz_angle_nonzero(self):
        # px=1, pz=0 → theta_xz should be 90°
        px, py, pz, p_mag = self._series(1.0, 0.0, 0.0, 1.0)
        _, _, _, theta_xz, _, _ = calculate_angles(px, py, pz, p_mag, "hd")
        assert float(theta_xz.iloc[0]) == pytest.approx(90.0, abs=1e-6)

    def test_hd_and_vd_give_different_rotated_angles(self):
        # Different wire angles → rotated projections must differ
        px, py, pz, p_mag = self._series(0.0, 1.0, 1.0, np.sqrt(2))
        _, theta_u_hd, _, _, _, _ = calculate_angles(px, py, pz, p_mag, "hd")
        _, theta_u_vd, _, _, _, _ = calculate_angles(px, py, pz, p_mag, "vd")
        assert float(theta_u_hd.iloc[0]) != pytest.approx(float(theta_u_vd.iloc[0]))

    def test_returns_six_values(self):
        px, py, pz, p_mag = self._series(0.0, 0.0, 1.0, 1.0)
        result = calculate_angles(px, py, pz, p_mag, "hd")
        assert len(result) == 6

    def test_unknown_detector_raises(self):
        px, py, pz, p_mag = self._series(0.0, 0.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            calculate_angles(px, py, pz, p_mag, "unknown")

    def test_all_angles_nonnegative(self):
        # After folding, all returned angles should be >= 0
        rng = np.random.default_rng(42)
        n = 20
        px = pd.Series(rng.normal(0, 1, n))
        py = pd.Series(rng.normal(0, 1, n))
        pz = pd.Series(rng.normal(0, 1, n))
        p_mag = pd.Series(np.sqrt(px**2 + py**2 + pz**2))
        for angle in calculate_angles(px, py, pz, p_mag, "vd"):
            assert (np.asarray(angle) >= -1e-10).all()
