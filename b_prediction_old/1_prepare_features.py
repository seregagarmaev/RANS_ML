import numpy as np
import pandas as pd
from modules.utils_old import generate_b, generate_S_R, calc_invariants, calc_tensor_basis, R_to_b_ij, wbRe
from scipy.interpolate import interp1d

cases = [180, 395, 550, 950, 1000, 2000, 5200]
nu = 0.0000232

for i, case in enumerate(cases):
    print(i+1, case)
    # prepare dns data
    dns_data = pd.read_feather(f'raw_data/{i+1}_dns.feather')
    b = generate_b(dns_data)
    dns_cell_centers = dns_data['y/delta'].values

    # prepare rans data
    grad_U = np.load(f'raw_data/{i+1}_rans_grad(U).npy')
    grad_U = grad_U.reshape(grad_U.shape[0], 3, 3)
    omega = np.load(f'raw_data/{i+1}_rans_omega.npy')
    k = np.load(f'raw_data/{i+1}_rans_k.npy')
    rans_cell_centers = np.load(f'raw_data/{i+1}_rans_Cy.npy')

    # perform interpolation
    b_interpolant = interp1d(dns_cell_centers, b, axis=0, kind='cubic')
    grad_U_interpolant = interp1d(rans_cell_centers, grad_U, axis=0, kind='cubic')
    omega_interpolant = interp1d(rans_cell_centers, omega, axis=0, kind='cubic')
    k_interpolant = interp1d(rans_cell_centers, k, axis=0, kind='cubic')
    new_space = np.geomspace(rans_cell_centers[0], dns_cell_centers[-1], 10000)
    # new_space = np.concatenate((new_space, np.linspace(0.01, dns_cell_centers[-1], 1000)))
    b_intd = b_interpolant(new_space)
    grad_U_intd = grad_U_interpolant(new_space)
    omega_intd = omega_interpolant(new_space)
    k_intd = k_interpolant(new_space)

    # calculate invariants and tensor basis
    S, R = generate_S_R(grad_U_intd, omega_intd)
    invariants = calc_invariants(S, R, 5)
    tensor_basis = calc_tensor_basis(S, R)
    wbRn_feature = wbRe(k_intd, new_space, nu)
    re_tau_feature = np.full(wbRn_feature.shape, case)


    # prepare RANS version of b_ij
    R_rans = np.load(f'raw_data/{i+1}_rans_turbulenceProperties:R.npy')
    b_rans = R_to_b_ij(R_rans)

    # save prepared target and features
    np.save(f'data/{i + 1}_inv.npy', invariants)
    np.save(f'data/{i + 1}_tb.npy', tensor_basis)
    np.save(f'data/{i + 1}_b.npy', b_intd)
    np.save(f'data/{i + 1}_wbRe.npy', wbRn_feature)
    np.save(f'data/{i + 1}_re_tau.npy', re_tau_feature)

    # save other data
    np.save(f'data/{i + 1}_space.npy', new_space)
    np.save(f'data/{i + 1}_b_RANS.npy', b_rans)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))
    for n in range(1, 6):
        plt.plot(new_space, invariants[:, n-1], label=f'{n}_invariant', alpha=0.5)
    # plt.plot(new_space, wbRn_feature, label='wall distance')
    plt.xscale('log')
    plt.legend()
    plt.title(f'{i} case')
    plt.savefig(f'plots/{i}_case.png', bbox_inches='tight')
