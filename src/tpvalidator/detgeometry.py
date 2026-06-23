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
    tpc_geo: tuple  # (n_drifts, n_y_rows, n_z_rows), e.g. (1, 8, 6)
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
    def num_drifts(self) -> int:
        return self.tpc_geo[0]

    @property
    def num_y_rows(self) -> int:
        return self.tpc_geo[1]

    @property
    def num_z_rows(self) -> int:
        return self.tpc_geo[2]

    @property
    def num_tpcs(self) -> int:
        return self.num_drifts * self.num_y_rows * self.num_z_rows
    

    @property
    def num_crps(self) -> float:
        return self.num_tpcs / 4

    def crp_num_chans_by_view_sim(self, ro_view: Literal[0, 1, 2]) -> int:
        """Return the simulated number of channels per CRP for a given readout view.

        Args:
            ro_view: Readout view index (0, 1, or 2).

        Returns:
            Number of CRP channels for the requested view.

        Raises:
            KeyError: If ro_view is not 0, 1, or 2.
        """
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
        """Return the simulated number of channels per TPC for a given readout view.

        Args:
            ro_view: Readout view index (0, 1, or 2).

        Returns:
            Number of TPC channels for the requested view.

        Raises:
            KeyError: If ro_view is not 0, 1, or 2.
        """
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
        """Convert a flat TPC ID to a 2-D grid position (j, k).

        The grid is indexed as (column, row) where j runs along the y-axis
        and k along the x-axis of the TPC layout.

        Args:
            tpc_id: Zero-based flat TPC index.

        Returns:
            Tuple (j, k) — grid coordinates of the TPC.
        """
        _, num_y, _ = self.tpc_geo
        k, j = divmod(tpc_id, num_y)
        return (j, k)

    def tpc_channel(self, channel):
        """Return the within-TPC channel index from a global channel number.

        Global channels encode the TPC index in the upper bits; this strips
        it and returns only the local channel offset.

        Args:
            channel: Global channel number.

        Returns:
            Zero-based channel index within the TPC.
        """
        _, tpc_ch = divmod(channel, self.tpc_tot_num_chans_sim)
        return tpc_ch

    def tpc_view_channel(self, channel):
        """Decompose a global channel number into (view, within-view channel index).

        Args:
            channel: Global channel number.

        Returns:
            Tuple (ro_view, view_channel) where ro_view is 0/1/2 and
            view_channel is the zero-based index within that view.
            Returns None if the channel is out of range.
        """
        tpc_ch = self.tpc_channel(channel)
        if tpc_ch < self.tpc_view_0_num_chans_sim:
            return (0, tpc_ch)
        elif tpc_ch < self.tpc_view_0_num_chans_sim + self.tpc_view_1_num_chans_sim:
            return (1, tpc_ch - self.tpc_view_0_num_chans_sim)
        elif tpc_ch < self.tpc_tot_num_chans_sim:
            return (2, tpc_ch - (self.tpc_view_0_num_chans_sim + self.tpc_view_1_num_chans_sim))

    def tpc_view_channel_range(self, ro_view: Literal[0, 1, 2]):
        """Return the half-open [start, stop) channel range for a readout view within a TPC.

        Args:
            ro_view: Readout view index (0, 1, or 2).

        Returns:
            Tuple (start, stop) of local TPC channel indices for the view.

        Raises:
            KeyError: If ro_view is not 0, 1, or 2.
        """
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

    def tpc_anode_surface(self) -> float:
        tpc = self.geo().tpcs[0]
        return tpc.y_range.length * tpc.z_range.length / 1e4


# Module-level singletons — use these instead of instantiating FDVDGeometry directly
FDVDGeometry_1x8x6  = FDVDGeometry(name='1x8x6', tpc_geo=(1, 8,  6), geo_resource="dunevd10kt_1x8x6_3view_30deg_geo.json")
FDVDGeometry_1x8x14 = FDVDGeometry(name='1x8x14', tpc_geo=(1, 8, 14), geo_resource="dunevd10kt_1x8x14_3view_30deg_geo.json")

_geo_map = {
   'dunevd10kt_3view_30deg_v5_refactored_1x8x14ref':  FDVDGeometry_1x8x14
}

def get_by_geocfg_id(geo_name:str):

    try:
        return _geo_map[geo_name]
    except KeyError as ke:
        raise KeyError(f'No detector geometry found matching {geo_name}')
    