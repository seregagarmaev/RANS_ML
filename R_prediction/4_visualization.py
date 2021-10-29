import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules.utils_old import generate_b, prepare_R, prepare_R_RANS


R_GB = np.load(f'data/predicted_R_TBGB.npy')
R_RF = np.load(f'data/predicted_R_TBRF.npy')
# b_NN = np.load(f'data/predicted_b_TBNN.npy')
R_rans = prepare_R_RANS(np.load(f'data/3_R_RANS.npy'))

nu = 0.0000998
grad_U_rans = 30.2011
u_tau_rans = np.sqrt(nu * grad_U_rans)
R_rans = R_rans / u_tau_rans / u_tau_rans


ls = np.load(f'data/3_space.npy')
cy_rans = np.load(f'raw_data/3_rans_Cy.npy')

dns_data = pd.read_feather(f'raw_data/3_dns.feather')
R_dns = prepare_R(dns_data)
dns_cell_centers = dns_data['y/delta'].values

fig, axs = plt.subplots(4, figsize=(10, 10), sharex=True)
for n, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1))):
    print(n, i, j)
    # figure = plt.figure(figsize=(10, 5))
    axs[n].plot(ls, R_GB[:, i, j], label='TBGB', linestyle='dashed')
    axs[n].plot(ls, R_RF[:, i, j], label='TBRF', linestyle='dashed')
    # axs[n].plot(ls, b_NN[:, i, j], label='TBNN', linestyle='dashed')
    axs[n].plot(cy_rans[:int(len(cy_rans)/2)], R_rans[:int(len(cy_rans)/2), i, j], label='RANS', linestyle='dashed')
    axs[n].plot(dns_cell_centers[1:], R_dns[1:, i, j], label='DNS', linestyle='dashed')
    # plt.yscale('log')
    # plt.xscale('log')
    axs[n].set_title(f"R_{i+1}{j+1} u'v' u'u' v'v'")
    axs[n].grid(color='grey', linestyle='--', linewidth=0.5)

axs[3].legend()
plt.xlabel('y/delta')
plt.xlim(cy_rans[0], 1)
# plt.xscale('log')
plt.savefig(f'plots/ML_R.png', bbox_inches='tight')