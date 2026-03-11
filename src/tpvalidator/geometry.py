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


    
    tpc_tot_num_chans_sim: int = 864
    tpc_view_0_num_chans_sim: int = 286 
    tpc_view_1_num_chans_sim: int = 286
    tpc_view_2_num_chans_sim: int = 292
    
    tpc_geo: tuple = (1,8,6)

    num_tpcs: int = tpc_geo[0]*tpc_geo[1]*tpc_geo[2]
    

    crp_tot_num_chans_sim: int = 3456
    crp_view_0_num_chans_sim: int = 1144 
    crp_view_1_num_chans_sim: int = 1144
    crp_view_2_num_chans_sim: int = 1168

    num_crps: int = num_tpcs/4


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

    @classmethod
    def tpc_num_chans_by_view_sim(cls, ro_view: int) -> int:

        match ro_view:
            case 0:
                return cls.tpc_view_0_num_chans_sim
            case 1:
                return cls.tpc_view_1_num_chans_sim
            case 2:
                return cls.tpc_view_2_num_chans_sim
            case _:
                raise KeyError(f"No {ro_view} readout view")
            

    @classmethod
    def tpc_id_to_grid(cls, tpc_id):

        _, num_y, _ = cls.tpc_geo

        k, j = divmod(tpc_id, num_y)
    
        return (j,k)

    @classmethod
    def tpc_channel(cls, channel):
        
        tpc_id, tpc_ch = divmod(channel, cls.tpc_tot_num_chans_sim)

        return tpc_ch

    @classmethod
    def tpc_view_channel(cls, channel):

        tpc_ch = cls.tpc_channel(channel)

        if tpc_ch < cls.tpc_view_0_num_chans_sim:
            return (0, tpc_ch)
        elif tpc_ch < cls.tpc_view_0_num_chans_sim+cls.tpc_view_1_num_chans_sim:
            return (1, tpc_ch - cls.tpc_view_0_num_chans_sim)
        elif tpc_ch < (cls.tpc_view_0_num_chans_sim+cls.tpc_view_1_num_chans_sim+cls.tpc_view_2_num_chans_sim):
            return (2, tpc_ch - (cls.tpc_view_0_num_chans_sim+cls.tpc_view_1_num_chans_sim))
            



