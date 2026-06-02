import json
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Literal


# ---------------------------------------------------------------------------
# Geometry element dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Range1D:
    min: float
    max: float

    @property
    def length(self) -> float:
        return self.max - self.min


@dataclass(frozen=True)
class BoxVolume:
    origin: Point3D
    x_range: Range1D
    y_range: Range1D
    z_range: Range1D

    @property
    def volume_m3(self) -> float:
        return self.x_range.length * self.y_range.length * self.z_range.length / 1e6


@dataclass(frozen=True)
class OpDet:
    origin: Point3D
    height: float
    length: float
    width: float

    @property
    def volume_m3(self) -> float:
        return self.height * self.length * self.width / 1e6


@dataclass(frozen=True)
class GeoData:
    detector_name: str
    cryostat: BoxVolume
    tpcs: tuple
    opdets: tuple


# ---------------------------------------------------------------------------
# JSON parsers
# ---------------------------------------------------------------------------

def _parse_point(d: dict) -> Point3D:
    return Point3D(x=d["x"], y=d["y"], z=d["z"])


def _parse_box(d: dict) -> BoxVolume:
    return BoxVolume(
        origin=_parse_point(d["origin"]),
        x_range=Range1D(min=d["x_range"]["min"], max=d["x_range"]["max"]),
        y_range=Range1D(min=d["y_range"]["min"], max=d["y_range"]["max"]),
        z_range=Range1D(min=d["z_range"]["min"], max=d["z_range"]["max"]),
    )


def _parse_geodata(d: dict) -> GeoData:
    return GeoData(
        detector_name=d["detector_name"],
        cryostat=_parse_box(d["cryostat"]),
        tpcs=tuple(_parse_box(t) for t in d["tpcs"]),
        opdets=tuple(
            OpDet(
                origin=_parse_point(o["origin"]),
                height=o["height"],
                length=o["length"],
                width=o["width"],
            )
            for o in d["opdets"]
        ),
    )


# ---------------------------------------------------------------------------
# FDVDGeometry
# ---------------------------------------------------------------------------

_geo_cache: dict = {}


@dataclass(frozen=True)
class FDVDGeometry:
    """VD detector geometry parameterised by TPC grid dimensions."""

    name: str
    tpc_geo: tuple  # (n_cryo, n_apa, n_tpc), e.g. (1, 8, 6)
    geo_resource: str | None = field(default=None, compare=False, hash=False)

    num_readout_views: int = 3
    num_readout_planes: int = 3

    # Per-view TPC channel counts
    tpc_view_0_num_chans_sim: int = 286
    tpc_view_1_num_chans_sim: int = 286
    tpc_view_2_num_chans_sim: int = 292

    # Per-view CRP channel counts
    crp_view_0_num_chans_sim: int = 1144
    crp_view_1_num_chans_sim: int = 1144
    crp_view_2_num_chans_sim: int = 1168

    @property
    def tpc_tot_num_chans_sim(self) -> int:
        return self.tpc_view_0_num_chans_sim + self.tpc_view_1_num_chans_sim + self.tpc_view_2_num_chans_sim

    @property
    def crp_tot_num_chans_sim(self) -> int:
        return self.crp_view_0_num_chans_sim + self.crp_view_1_num_chans_sim + self.crp_view_2_num_chans_sim

    @property
    def num_tpcs(self) -> int:
        return self.tpc_geo[0] * self.tpc_geo[1] * self.tpc_geo[2]

    @property
    def num_crps(self) -> float:
        return self.num_tpcs / 4

    def crp_num_chans_by_view_sim(self, ro_view: Literal[0, 1, 2]) -> int:
        match ro_view:
            case 0:
                return self.crp_view_0_num_chans_sim
            case 1:
                return self.crp_view_1_num_chans_sim
            case 2:
                return self.crp_view_2_num_chans_sim
            case _:
                raise KeyError(f"No {ro_view} readout view")

    def tpc_num_chans_by_view_sim(self, ro_view: Literal[0, 1, 2]) -> int:
        match ro_view:
            case 0:
                return self.tpc_view_0_num_chans_sim
            case 1:
                return self.tpc_view_1_num_chans_sim
            case 2:
                return self.tpc_view_2_num_chans_sim
            case _:
                raise KeyError(f"No {ro_view} readout view")

    def tpc_id_to_grid(self, tpc_id):
        _, num_y, _ = self.tpc_geo
        k, j = divmod(tpc_id, num_y)
        return (j, k)

    def tpc_channel(self, channel):
        _, tpc_ch = divmod(channel, self.tpc_tot_num_chans_sim)
        return tpc_ch

    def tpc_view_channel(self, channel):
        tpc_ch = self.tpc_channel(channel)
        if tpc_ch < self.tpc_view_0_num_chans_sim:
            return (0, tpc_ch)
        elif tpc_ch < self.tpc_view_0_num_chans_sim + self.tpc_view_1_num_chans_sim:
            return (1, tpc_ch - self.tpc_view_0_num_chans_sim)
        elif tpc_ch < self.tpc_tot_num_chans_sim:
            return (2, tpc_ch - (self.tpc_view_0_num_chans_sim + self.tpc_view_1_num_chans_sim))

    def tpc_view_channel_range(self, ro_view: Literal[0, 1, 2]):
        match ro_view:
            case 0:
                return (0, self.tpc_view_0_num_chans_sim)
            case 1:
                return (self.tpc_view_0_num_chans_sim, self.tpc_view_0_num_chans_sim + self.tpc_view_1_num_chans_sim)
            case 2:
                return (self.tpc_view_0_num_chans_sim + self.tpc_view_1_num_chans_sim, self.tpc_tot_num_chans_sim)
            case _:
                raise KeyError(f"No {ro_view} readout view")

    def geo(self) -> GeoData:
        """Return the full geometry loaded from the bundled JSON resource."""
        if self not in _geo_cache:
            if self.geo_resource is None:
                raise ValueError("No geo_resource set for this FDVDGeometry instance")
            text = files("tpvalidator.data.geo").joinpath(self.geo_resource).read_text()
            _geo_cache[self] = _parse_geodata(json.loads(text))
        return _geo_cache[self]

    def cryo_volume(self) -> float:
        return self.geo().cryostat.volume_m3

    def tpc_volume(self) -> float:
        return self.geo().tpcs[0].volume_m3

    def det_volume(self) -> float:
        return self.tpc_volume() * self.num_tpcs

    def anode_surface(self) -> float:
        tpc = self.geo().tpcs[0]
        return tpc.y_range.length * tpc.z_range.length / 1e4


# Module-level singletons — use these instead of instantiating FDVDGeometry directly
FDVDGeometry_1x8x6  = FDVDGeometry(name='1x8x6', tpc_geo=(1, 8,  6), geo_resource="dunevd10kt_1x8x6_3view_30deg_geo.json")
FDVDGeometry_1x8x14 = FDVDGeometry(name='1x8x14', tpc_geo=(1, 8, 14), geo_resource="dunevd10kt_1x8x14_3view_30deg_geo.json")
