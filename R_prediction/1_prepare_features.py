import numpy as np
import pandas as pd
from modules.utils_old import generate_b, generate_S_R, calc_invariants, calc_tensor_basis, R_to_b_ij, prepare_R, prepare_R_RANS
from scipy.interpolate import interp1d

cases = [180, 395, 550, 950, 1000, 2000, 5200]


for i, case in enumerate(cases):
    print(i+1, case)
    # prepare dns data
    dns_data = pd.read_feather(f'raw_data/{i+1}_dns.feather')
    Rij = prepare_R(dns_data)
    dns_cell_centers = dns_data['y/delta'].values

    # prepare rans data
    grad_U = np.load(f'raw_data/{i+1}_rans_grad(U).npy')
    grad_U = grad_U.reshape(grad_U.shape[0], 3, 3)
    omega = np.load(f'raw_data/{i+1}_rans_omega.npy')
    rans_cell_centers = np.load(f'raw_data/{i+1}_rans_Cy.npy')

    # perform interpolation
    Rij_interpolant = interp1d(dns_cell_centers, Rij, axis=0, kind='cubic')
    grad_U_interpolant = interp1d(rans_cell_centers, grad_U, axis=0, kind='cubic')
    omega_interpolant = interp1d(rans_cell_centers, omega, axis=0, kind='cubic')
    new_space = np.geomspace(rans_cell_centers[0], dns_cell_centers[-1], 1000)
    Rij_intd = Rij_interpolant(new_space)
    grad_U_intd = grad_U_interpolant(new_space)
    omega_intd = omega_interpolant(new_space)

    # calculate invariants and tensor basis
    S, R = generate_S_R(grad_U_intd, omega_intd)
    invariants = calc_invariants(S, R, 5)
    tensor_basis = calc_tensor_basis(S, R)

    # prepare RANS version of R_ij
    R_rans = np.load(f'raw_data/{i+1}_rans_turbulenceProperties:R.npy')

    # save prepared target and features
    np.save(f'data/{i + 1}_inv.npy', invariants)
    np.save(f'data/{i + 1}_tb.npy', tensor_basis)
    np.save(f'data/{i + 1}_R.npy', Rij_intd)

    # save other data
    np.save(f'data/{i + 1}_space.npy', new_space)
    np.save(f'data/{i + 1}_R_RANS.npy', R_rans)
