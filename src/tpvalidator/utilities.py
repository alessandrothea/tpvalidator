import uproot
import awkward as ak
import pandas as pd
import numpy as np

def load_data(file_path: str, tree_name: str = 'triggerana/tree', branch_names: list = None, max_events=None) -> pd.DataFrame:
    """
    Loads data from a ROOT tree into a Pandas Dataframe after expanding vectors into rows.

    Args:
        file_path (str): path to the root file containg the ROOT tree
        branch_names (list): _description_
        max_events (int, optional): _description_. Defaults to 1000.

    Returns:
        pd.DataFrame: _description_
    """
    try:
        # with uproot.open(file_path) as file:
        #     tree = file["triggerana/tree"]
        #     arrays = tree.arrays(branch_names, library="ak", entry_stop=max_events)
        #     return ak.to_dataframe(arrays)
        with uproot.open(f'{file_path}:{tree_name}') as tree:
            arrays = tree.arrays(branch_names, library="ak", entry_stop=max_events)
            return ak.to_dataframe(arrays)
    
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None
    


def calculate_angles(px, py, pz, p_mag):
    """
    Calculate:
    θ_y (angle w.r.t vertical y-axis --> *should* align with collection plane orientation in hd),
    θ_U (angle w.r.t U-plane wires at +37.5°),
    θ_V (angle w.r.t V-plane wires at -37.5°),
    Based on momentum of the MC particles in x,y,z directions 
        
    Returns:
    theta_y, theta_U, theta_V, theta_xz, theta_xz_U, theta_xz_V (all in degrees)
    """
    # θ_y: Angle relative to vertical y-axis, where θ_y = 0 is upward
    #theta_y = np.degrees( np.arcsin(py / p_mag))
    theta_y = np.degrees(np.arccos(py / p_mag))
    #theta_y = (theta_y + 180) % 360 - 180
    
    # θ_xz: Angle in the xz-plane (measured from z-axis)
    theta_xz = np.degrees(np.arctan2(px, pz))
    
    # Rotation angles for U and V planes (±37.5 degrees in zy-plane)
    theta_rot_U = np.radians(-37.5)  # U-plane rotation
    theta_rot_V = np.radians(37.5)   # V-plane rotation
    
    # Rotate momentum components in zy-plane for U-plane
    p_y_U = py * np.cos(theta_rot_U) - pz * np.sin(theta_rot_U)
    p_z_U = py * np.sin(theta_rot_U) + pz * np.cos(theta_rot_U)
    theta_xz_U = np.degrees(np.arctan2(px, p_z_U))
    theta_y_U =  90 - np.degrees(np.arcsin(p_y_U / p_mag))
    #theta_y_U = (theta_y_U + 180) % 360 - 180 
 
    # Rotate momentum components in zy-plane for V-plane
    p_y_V = py * np.cos(theta_rot_V) - pz * np.sin(theta_rot_V)
    p_z_V = py * np.sin(theta_rot_V) + pz * np.cos(theta_rot_V)
    theta_xz_V = np.degrees(np.arctan2(px, p_z_V))
    theta_y_V = 90-np.degrees(np.arcsin(p_y_V / p_mag))
    #theta_y_V = (theta_y_V + 180) % 360 - 180

    return theta_y, theta_y_U, theta_y_V, abs(theta_xz), abs(theta_xz_U), abs(theta_xz_V)

# Enable when ready
#@njit(cache=True, nogil=True)
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def wrap_phi(a):
    conds = [
        a > 90,
        a < -90,
    ]

    choices = [
        a - 180,
        a + 180
    ]
    return abs(a)
    return np.select(conds, choices, default=a)


def calculate_angles_2(px, py, pz, p_mag):
    """
    Calculate:

    1. angle between p and drift direction
    2. angle between p and collection wire direction
    2. angle between p and induction wires direction


    Note about VD geometry
    - z is the direction of the beam
    - The drift direction is upward/downward : x, in vd coordinates
    - y is the direction of collection strips on plane X (hd) or Z (vd)
    - u, the direction of the first induction plane strips is (0, -0.5, 0.866) in LArSoft
    - v, the direction of the first induction plane strips is (0, -0.5, -0.866) in LArSoft


    θ_y (angle w.r.t vertical y-axis --> *should* align with collection plane orientation in hd),
    θ_U (angle w.r.t U-plane wires at +37.5° in hd),
    θ_V (angle w.r.t V-plane wires at -37.5°),
    Based on momentum of the MC particles in x,y,z directions 
        
    Returns:
    theta_drift, theta_beam, theta_coll, theta_u, theta_v, phi_coll, phi_ind_u, phi_ind_v (all in degrees)
    """

    angle_ind  = np.radians(-30) # degrees
    sin_ind  = np.sin(angle_ind)
    cos_ind  = np.cos(angle_ind)
    # unitary vectors for each plane
    e_coll = [0,1,0]

    # Define induction wires vector basis
    e_ind_u = [0, sin_ind, cos_ind]
    k_ind_u = [0, -cos_ind, sin_ind]

    # e_ind_v = [0, sin_ind, -cos_ind]
    # k_ind_v = [0, cos_ind, sin_ind]

    # angle_ind  = np.radians(-150) # degrees
    angle_ind  = np.radians(30) # degrees
    sin_ind  = np.sin(angle_ind)
    cos_ind  = np.cos(angle_ind)
    e_ind_v = [0, sin_ind, cos_ind]
    k_ind_v = [0, -cos_ind, sin_ind]

    # Calculate prokections on all axes
    pe_ind_u = e_ind_u[1]*py+e_ind_u[2]*pz
    pk_ind_u = k_ind_u[1]*py+k_ind_u[2]*pz

    pe_ind_v = e_ind_v[1]*py+e_ind_v[2]*pz
    pk_ind_v = k_ind_v[1]*py+k_ind_v[2]*pz


    theta_u = np.degrees(np.arccos(pe_ind_u / p_mag))
    theta_v = np.degrees(np.arccos(pe_ind_v / p_mag))

    phi_coll = wrap_phi(np.degrees(np.arctan2 ( pz      , px )))
    phi_ind_u = wrap_phi(np.degrees(np.arctan2( pk_ind_u, px )))
    phi_ind_v = wrap_phi(np.degrees(np.arctan2( pk_ind_v, px )))

    theta_drift = np.degrees(np.arccos(px / p_mag))
    theta_coll = np.degrees(np.arccos(py / p_mag))
    theta_beam = np.degrees(np.arccos(pz / p_mag))

    return theta_drift, theta_beam, theta_coll, theta_u, theta_v, phi_coll, phi_ind_u, phi_ind_v