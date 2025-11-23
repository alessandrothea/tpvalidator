
# Number of channels per CRP in simulation
crp_tot_num_chans_sim = 3456
crp_view_0_num_chans_sim = 1144 
crp_view_1_num_chans_sim = 1144
crp_view_2_num_chans_sim = 1168

crp_num_chans_by_view_sim: dict = {
    0: crp_view_0_num_chans_sim,
    1: crp_view_1_num_chans_sim,
    2: crp_view_2_num_chans_sim,
}

from dataclasses import dataclass

@dataclass(frozen=True)
class FDVDGeometry_1x6x8:

    crp_tot_num_chans_sim: int = 3456
    crp_view_0_num_chans_sim: int = 1144 
    crp_view_1_num_chans_sim: int = 1144
    crp_view_2_num_chans_sim: int = 1168

    num_crps: int = 12
    num_tpc: int = num_crps*4

    @classmethod
    def crp_num_chans_by_view_sim(cls, ro_view: int) -> int:

        match ro_view:
            case 0:
                return cls.crp_view_0_num_chans_sim
            case 1:
                return cls.crp_view_1_num_chans_sim
            case 2:
                return cls.crp_view_2_num_chans_sim
            case _:
                raise KeyError(f"No {ro_view} readout view")