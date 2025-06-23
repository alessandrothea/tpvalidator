import numpy as np

from tpvalidator.utilities import calculate_angles_2
def test_angles():
    

    px = np.array([1,0,0])
    py = np.array([0,1,0])
    pz = np.array([0,0,1])

    p = np.stack((px, py, pz), axis=-1)

    p_mag = np.linalg.norm(p, axis=1)
    print(p_mag)

    theta_drift, theta_beam, theta_coll, phi_coll, phi_ind_u, phi_ind_v = calculate_angles_2(px,py,pz,p_mag)


    print(f"theta_drift = {theta_drift}")
    print(f"theta_coll = {theta_coll}")
    print(f"theta_beam = {theta_beam}")
    print(f"phi_coll = {phi_coll}")
    print(f"phi_ind_u = {phi_ind_u}")
    print(f"phi_ind_v = {phi_ind_v}")




if __name__ == '__main__':
    test_angles()
