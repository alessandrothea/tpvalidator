import pytest

from tpvalidator.detector_geometry import (
    Point3D,
    Range1D,
    BoxVolume,
    FDVDGeometry,
    FDVDGeometry_1x8x6,
    FDVDGeometry_1x8x14,
)


class TestRange1D:
    def test_length(self):
        r = Range1D(min=2.0, max=7.0)
        assert r.length == pytest.approx(5.0)

    def test_zero_length(self):
        r = Range1D(min=3.0, max=3.0)
        assert r.length == pytest.approx(0.0)


class TestBoxVolume:
    def _cube(self, side_cm):
        o = Point3D(0, 0, 0)
        r = Range1D(0, side_cm)
        return BoxVolume(origin=o, x_range=r, y_range=r, z_range=r)

    def test_unit_cube_volume(self):
        # 100 cm × 100 cm × 100 cm = 1 m³
        box = self._cube(100.0)
        assert box.volume_m3 == pytest.approx(1.0)

    def test_volume_scales_cubically(self):
        box_1 = self._cube(100.0)
        box_2 = self._cube(200.0)
        assert box_2.volume_m3 == pytest.approx(box_1.volume_m3 * 8)


class TestFDVDGeometryChannelMath:
    geo = FDVDGeometry_1x8x6  # tpc_geo=(1,8,6); view counts: 286, 286, 292

    def test_total_channels(self):
        assert self.geo.tpc_tot_num_chans_sim == 286 + 286 + 292

    def test_crp_total_channels(self):
        assert self.geo.crp_tot_num_chans_sim == 1144 + 1144 + 1168

    def test_num_tpcs(self):
        assert self.geo.num_tpcs == 1 * 8 * 6

    def test_tpc_num_chans_by_view(self):
        assert self.geo.tpc_num_chans_by_view_sim(0) == 286
        assert self.geo.tpc_num_chans_by_view_sim(1) == 286
        assert self.geo.tpc_num_chans_by_view_sim(2) == 292

    def test_tpc_num_chans_invalid_view(self):
        with pytest.raises(KeyError):
            self.geo.tpc_num_chans_by_view_sim(3)

    def test_tpc_view_channel_range_view0(self):
        lo, hi = self.geo.tpc_view_channel_range(0)
        assert lo == 0
        assert hi == 286

    def test_tpc_view_channel_range_view2(self):
        lo, hi = self.geo.tpc_view_channel_range(2)
        assert lo == 286 + 286
        assert hi == 286 + 286 + 292

    def test_tpc_view_channel_range_non_empty(self):
        for v in (0, 1, 2):
            lo, hi = self.geo.tpc_view_channel_range(v)
            assert hi > lo


class TestFDVDGeometryGrid:
    geo = FDVDGeometry_1x8x6  # num_y=8

    def test_first_tpc(self):
        assert self.geo.tpc_id_to_grid(0) == (0, 0)

    def test_second_tpc(self):
        assert self.geo.tpc_id_to_grid(1) == (1, 0)

    def test_next_row(self):
        assert self.geo.tpc_id_to_grid(8) == (0, 1)

    def test_1x8x14_grid(self):
        # num_y=8 same, just more in z
        assert FDVDGeometry_1x8x14.tpc_id_to_grid(0) == (0, 0)
        assert FDVDGeometry_1x8x14.tpc_id_to_grid(8) == (0, 1)


class TestFDVDGeometryGeoLoad:
    def test_geo_loads_without_error(self):
        geo_data = FDVDGeometry_1x8x6.geo()
        assert geo_data is not None

    def test_tpcs_non_empty(self):
        geo_data = FDVDGeometry_1x8x6.geo()
        assert len(geo_data.tpcs) > 0

    def test_correct_num_tpcs(self):
        geo_data = FDVDGeometry_1x8x6.geo()
        assert len(geo_data.tpcs) == FDVDGeometry_1x8x6.num_tpcs

    def test_geo_cached(self):
        # Two calls return the same object
        assert FDVDGeometry_1x8x6.geo() is FDVDGeometry_1x8x6.geo()

    def test_no_geo_resource_raises(self):
        # Use a tpc_geo tuple not shared with any module-level singleton so the
        # cache lookup misses and the None-resource guard is actually exercised.
        g = FDVDGeometry(tpc_geo=(99, 99, 99), geo_resource=None)
        with pytest.raises(ValueError):
            g.geo()
