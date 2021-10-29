import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from modules.utils import *
from config import *


for i, re_t in enumerate(re_t_list):
    ########################### READING RANS ###################################
    rans_cc = read_field('C', max_step, rans_path + f'/{i+1}_pc_Re_tau_{re_t}/')

    # here we take only cells along y axis with x = 0.025 and z = 0.025
    c1 = rans_cc[:, 0] == 0.025
    c2 = rans_cc[:, 2] == 0.025
    mask = c1 & c2
    rans_cy = rans_cc[mask, 1]

    fields = {}
    for field_name in field_names:
        fields[field_name] = read_field(field_name, max_step, rans_path + f'/{i+1}_pc_Re_tau_{re_t}/')[mask]

    ########################### READING DNS ###################################
    dns_data = pd.read_csv(f'{dns_path}/{i + 1}_pc_Re_tau_{re_t}/half_channel_data.csv')[dns_columns]
    R_dns = dns_to_R(dns_data)
    b_dns = R_to_b(R_dns)
    dns_cy = dns_data['y/delta'].values

    ########################### INTERPOLATION ###################################
    b_interpolant = interp1d(dns_cy, b_dns, axis=0, kind='cubic')
    grad_U_interpolant = interp1d(rans_cy, fields['grad(U)'], axis=0, kind='cubic')
    omega_interpolant = interp1d(rans_cy, fields['omega'], axis=0, kind='cubic')
    k_interpolant = interp1d(rans_cy,fields['k'], axis=0, kind='cubic')
    new_space = np.geomspace(rans_cy[0], dns_cy[-1], 1000)

    b_intd = b_interpolant(new_space)
    grad_U_intd = grad_U_interpolant(new_space)
    omega_intd = omega_interpolant(new_space)
    k_intd = k_interpolant(new_space)

    ########################### CALCULATE S AND R ###################################
    S, R = generate_S_R(grad_U_intd, omega_intd)
    invariants = calc_invariants(S, R)
    tensor_basis = calc_tensor_basis(S, R)
    wbReNum = wbRe(k_intd, new_space, nus[i])

























    # import matplotlib.pyplot as plt
    #
    # b_rans = R_to_b(fields['turbulenceProperties:R'])
    #
    # plt.figure(figsize=(10, 7))
    # plt.plot(dns_cy, b_dns[:, 0, 1], label='DNS', linestyle='dashed', alpha=0.5)
    # plt.plot(rans_cy, b_rans[:, 0, 1], label='RANS', linestyle='dashed', alpha=0.5)
    # plt.plot(new_space, b_intd[:, 0, 1], label='interpolated', linestyle='dashed', alpha=0.7)
    # plt.xscale('log')
    # plt.legend()
    # plt.savefig('b.PNG', bbox_inches='tight')
    #
    # print(rans_cy)





    # print(new_space[])
    # new_space = np.logspace(rans_cy[0], dns_cell_centers[-1], 1000)
    # b_intd = b_interpolant(new_space)
    # grad_U_intd = grad_U_interpolant(new_space)
    # omega_intd = omega_interpolant(new_space)




    break