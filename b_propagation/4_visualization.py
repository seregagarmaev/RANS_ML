import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# constants
nu = 0.0000998
grad_U_rans = 31.0798
grad_U_ml = 17.1331
grad_U_rans_dns = 19.2866
n_cells = 640

u_tau_rans = np.sqrt(nu * grad_U_rans)
u_tau_ml = np.sqrt(nu * grad_U_ml)
u_tau_rans_dns = np.sqrt(nu * grad_U_rans_dns)
u_tau_dns = 5.43496e-02

# load rans data
U_rans = np.load('data/U_rans.npy')[:int(n_cells/2), 0]
k_rans = np.load('data/k_rans.npy')[:int(n_cells/2)] / (u_tau_rans ** 2)
omega_rans = np.load('data/omega_rans.npy')[:int(n_cells/2)]

# load rans+ml data
U_rans_ml = np.load('data/U_rans_ml.npy')[:int(n_cells/2), 0]
k_rans_ml = np.load('data/k_rans_ml.npy')[:int(n_cells/2)] / (u_tau_ml ** 2)
omega_rans_ml = np.load('data/omega_rans_ml.npy')[:int(n_cells/2)]

# load rans+dns data
U_rans_dns = np.load('data/U_rans_dns.npy')[:int(n_cells/2), 0]
k_rans_dns = np.load('data/k_rans_dns.npy')[:int(n_cells/2)] / (u_tau_ml ** 2)
omega_rans_dns = np.load('data/omega_rans_dns.npy')[:int(n_cells/2)]

# load dns data
dns = pd.read_csv('../b_prediction_old/DNS_cases/3_pc_Re_tau_550/mean_velocity_profile.csv')
dns_k = pd.read_csv('../b_prediction_old/DNS_cases/3_pc_Re_tau_550/half_channel_data.csv')

# prepare normalized U fields
Uplus_rans = U_rans / u_tau_rans
Uplus_rans_ml = U_rans_ml / u_tau_ml
Uplus_rans_dns = U_rans_dns / u_tau_rans_dns
Uplus_dns = dns['U']

# denormalize U+ from DNS
U_dns = Uplus_dns * u_tau_dns

# prepare k field from dns
k_dns = dns_k['k']

# prepare cell center coordinates
cy_rans = np.load('data/Cy_rans.npy')[:int(n_cells/2)]
cy_dns = dns['y/delta']
yplus = dns['y^+']
yplus_rans = cy_rans * u_tau_rans / nu
yplus_rans_ml = cy_rans * u_tau_ml / nu
yplus_rans_dns = cy_rans * u_tau_rans_dns / nu



# plot viscous velocity plot
plt.figure(figsize=(10,7))
plt.plot(cy_rans, U_rans, label='RANS', linestyle='dashed')
plt.plot(cy_rans, U_rans_ml, label='RANS+ML', linestyle='dashed')
plt.plot(cy_rans, U_rans_dns, label='RANS+DNS', linestyle='dashed')
plt.plot(cy_dns, U_dns, label='DNS', linestyle='dashed')
plt.xscale('log')
plt.legend(loc='lower right')
# plt.title('Mean velocity')
plt.xlim(0.001, 1)
plt.xlabel('y/delta')
plt.ylabel('U / U_b')
plt.grid(color='grey', linestyle='--', linewidth=0.5)
plt.savefig(f'plots/mean_velocity.png', bbox_inches='tight')

# plot turbulent kinetic energy plot
plt.figure(figsize=(10,7))
plt.plot(cy_rans, k_rans, label='RANS', linestyle='dashed')
plt.plot(cy_rans, k_rans_ml, label='RANS+ML', linestyle='dashed')
plt.plot(cy_rans, k_rans_dns, label='RANS+DNS', linestyle='dashed')
plt.plot(cy_dns, k_dns, label='DNS', linestyle='dashed')
# plt.yscale('log')
plt.xscale('log')
plt.xlabel('y\delta')
plt.ylabel('k')
plt.xlim(0.001, 1)
plt.legend(loc='lower right')
plt.title('Turbulent kinetic energy')
plt.grid(color='grey', linestyle='--', linewidth=0.5)
plt.savefig(f'plots/turbulent_kinetic_energy.png', bbox_inches='tight')









# # plot viscous velocity plot
# plt.figure(figsize=(10,5))
# plt.plot(yplus_rans, Uplus_rans, label='RANS')
# plt.plot(yplus_rans_ml, Uplus_rans_ml, label='RANS+ML')
# plt.plot(yplus_rans_dns, Uplus_rans_dns, label='RANS+DNS')
# plt.plot(yplus, Uplus_dns, label='DNS')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend(loc='lower right')
# # plt.title('Mean velocity')
# plt.xlabel('y+')
# plt.ylabel('U+')
# plt.grid(color='grey', linestyle='--', linewidth=0.5)
# plt.savefig(f'plots/mean_velocity_yplus.png', bbox_inches='tight')

# plot turbulent kinetic enegry plot
plt.figure(figsize=(10,5))
plt.plot(yplus_rans, k_rans, label='RANS')
plt.plot(yplus_rans_ml, k_rans_ml, label='RANS+ML')
plt.plot(yplus_rans_dns, k_rans_dns, label='RANS+DNS')
plt.plot(yplus, k_dns, label='DNS')
# plt.yscale('log')
plt.xscale('log')
plt.xlabel('y+')
plt.ylabel('k')
plt.xlim(0.1, 1000)
plt.legend(loc='lower right')
plt.grid(color='grey', linestyle='--', linewidth=0.5)
plt.savefig(f'plots/turbulent_kinetic_energy_yplus.png', bbox_inches='tight')
