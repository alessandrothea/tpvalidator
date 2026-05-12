import json
from dataclasses import dataclass, field
from importlib.resources import files

_geo_cache: dict = {}


@dataclass(frozen=True)
class FDVDGeometry:
    """VD detector geometry parameterised by TPC grid dimensions."""

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

    def crp_num_chans_by_view_sim(self, ro_view: int) -> int:
        match ro_view:
            case 0:
                return self.crp_view_0_num_chans_sim
            case 1:
                return self.crp_view_1_num_chans_sim
            case 2:
                return self.crp_view_2_num_chans_sim
            case _:
                raise KeyError(f"No {ro_view} readout view")

    def tpc_num_chans_by_view_sim(self, ro_view: int) -> int:
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
        tpc_id, tpc_ch = divmod(channel, self.tpc_tot_num_chans_sim)
        return tpc_ch

    def tpc_view_channel(self, channel):
        tpc_ch = self.tpc_channel(channel)
        if tpc_ch < self.tpc_view_0_num_chans_sim:
            return (0, tpc_ch)
        elif tpc_ch < self.tpc_view_0_num_chans_sim + self.tpc_view_1_num_chans_sim:
            return (1, tpc_ch - self.tpc_view_0_num_chans_sim)
        elif tpc_ch < self.tpc_tot_num_chans_sim:
            return (2, tpc_ch - (self.tpc_view_0_num_chans_sim + self.tpc_view_1_num_chans_sim))
        
    def geo(self) -> dict:
        """Return the full geometry dict loaded from the bundled JSON resource."""
        if self not in _geo_cache:
            if self.geo_resource is None:
                raise ValueError("No geo_resource set for this FDVDGeometry instance")
            text = files("tpvalidator.data.geo").joinpath(self.geo_resource).read_text()
            _geo_cache[self] = json.loads(text)
        return _geo_cache[self]

    def tpc_view_channel_range(self, ro_view):
        match ro_view:
            case 0:
                return (0, self.tpc_view_0_num_chans_sim)
            case 1:
                return (self.tpc_view_0_num_chans_sim, self.tpc_view_0_num_chans_sim+self.tpc_view_1_num_chans_sim)
            case 2:
                return (self.tpc_view_0_num_chans_sim+self.tpc_view_1_num_chans_sim, self.tpc_view_0_num_chans_sim+self.tpc_view_1_num_chans_sim+self.tpc_view_2_num_chans_sim)
            case _:
                raise KeyError(f"No {ro_view} readout view")

# Module-level singletons — use these instead of instantiating FDVDGeometry directly
FDVDGeometry_1x8x6  = FDVDGeometry(tpc_geo=(1, 8,  6), geo_resource="dunevd10kt_1x8x6_3view_30deg_geo.json")
FDVDGeometry_1x8x14 = FDVDGeometry(tpc_geo=(1, 8, 14), geo_resource="dunevd10kt_1x8x14_3view_30deg_geo.json")
