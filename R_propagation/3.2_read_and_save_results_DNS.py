import numpy as np
from modules.utils_old import read_field


rans_max_step = 10000
dns_max_step = 14168
mesh_size = 640
n_cells_along_x = 2


# load and save rans+dns fields
U_rans_ml = read_field(f'3_pc_Re_tau_550_DNS/{dns_max_step}/U',mesh_size, n_cells_along_x)
k_rans_ml = read_field(f'3_pc_Re_tau_550_DNS/{dns_max_step}/k',mesh_size, n_cells_along_x)
omega_rans_ml = read_field(f'3_pc_Re_tau_550_DNS/{dns_max_step}/omega',mesh_size, n_cells_along_x)
np.save('data/U_rans_dns.npy', U_rans_ml)
np.save('data/k_rans_dns.npy', k_rans_ml)
np.save('data/omega_rans_dns.npy', omega_rans_ml)