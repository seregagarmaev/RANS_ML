import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules.utils_old import generate_b


b_GB = np.load(f'data/predicted_b_TBGB_filtered.npy')
b_RF = np.load(f'data/predicted_b_TBRF_filtered.npy')
# b_NN = np.load(f'data/predicted_b_TBNN_filtered.npy')
b_rans = np.load(f'data/3_b_RANS.npy')

ls = np.load(f'data/3_space.npy')
cy_rans = np.load(f'raw_data/3_rans_Cy.npy')

dns_data = pd.read_feather(f'raw_data/3_dns.feather')
b_dns = generate_b(dns_data)
dns_cell_centers = dns_data['y/delta'].values

fig, axs = plt.subplots(4, figsize=(7, 10), sharex=True)
for n, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1))):
    print(n, i, j)
    # figure = plt.figure(figsize=(10, 5))
    axs[n].plot(ls, b_GB[:, i, j], label='TBGB', linestyle='dashed')
    axs[n].plot(ls, b_RF[:, i, j], label='TBRF', linestyle='dashed')
    # axs[n].plot(ls, b_NN[:, i, j], label='TBNN', linestyle='dashed')
    axs[n].plot(cy_rans[:int(len(cy_rans)/2)], b_rans[:int(len(cy_rans)/2), i, j], label='RANS', linestyle='dashed')
    axs[n].plot(dns_cell_centers[1:], b_dns[1:, i, j], label='DNS', linestyle='dashed')
    # plt.yscale('log')
    plt.xscale('log')
    # plt.title(f'Reynolds stress anisotropy b_{i+1}{j+1}')
    axs[n].set_title(f'b_{i+1}{j+1}')
    axs[n].grid(color='grey', linestyle='--', linewidth=0.5)

axs[3].legend()
plt.xlabel('y/delta')
plt.xlim(cy_rans[0], 1)
# plt.xscale('log')
plt.savefig(f'plots/ML_b_filtered.png', bbox_inches='tight')