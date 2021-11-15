import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from modules.utils import *
from config import *

X_test, X_train = [], []
TB_test, TB_train = [], []
b_train, b_test = [], []

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

    b_rans = R_to_b(fields['turbulenceProperties:R'])

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
    re_tau_feature = np.full(wbReNum.shape, re_t)

    # merge features
    wbReNum = wbReNum.reshape((-1, 1))
    re_tau_feature = re_tau_feature.reshape((-1, 1))
    features = np.concatenate((invariants, wbReNum, re_tau_feature), axis=1)

    # save prepared target and features
    np.save(f'data/{i + 1}_inv.npy', invariants)
    np.save(f'data/{i + 1}_tb.npy', tensor_basis)
    np.save(f'data/{i + 1}_b.npy', b_intd)
    np.save(f'data/{i + 1}_wbRe.npy', wbReNum)
    np.save(f'data/{i + 1}_re_tau.npy', re_tau_feature)

    # save other data
    np.save(f'data/{i + 1}_space.npy', new_space)
    np.save(f'data/{i + 1}_b_RANS.npy', b_rans)

    # prepare train and test data
    if re_t == test_case:
        X_test.append(features)
        TB_test.append(tensor_basis)
        b_test.append(b_intd)
    else:
        X_train.append(features)
        TB_train.append(tensor_basis)
        b_train.append(b_intd)

X_test, X_train = np.concatenate(X_test), np.concatenate(X_train)
TB_test, TB_train = np.concatenate(TB_test), np.concatenate(TB_train)
b_test, b_train = np.concatenate(b_test), np.concatenate(b_train)

np.save(f'data/X_test.npy', X_test)
np.save(f'data/X_train.npy', X_train)
np.save(f'data/TB_test.npy', TB_test)
np.save(f'data/TB_train.npy', TB_train)
np.save(f'data/b_test.npy', b_test)
np.save(f'data/b_train.npy', b_train)
