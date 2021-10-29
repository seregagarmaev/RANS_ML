import numpy as np
from modules.utils_old import read_field


rans_max_step = 10000
ml_max_step = 13828.6
mesh_size = 640
n_cells_along_x = 2

# load and save rans fields
U_rans = read_field(f'3_pc_Re_tau_550/{rans_max_step}/U', mesh_size, n_cells_along_x)
k_rans = read_field(f'3_pc_Re_tau_550/{rans_max_step}/k', mesh_size, n_cells_along_x)
omega_rans = read_field(f'3_pc_Re_tau_550/{rans_max_step}/omega', mesh_size, n_cells_along_x)
cy = read_field(f'3_pc_Re_tau_550/{rans_max_step}/Cy', mesh_size, n_cells_along_x)
np.save('data/U_rans.npy', U_rans)
np.save('data/k_rans.npy', k_rans)
np.save('data/omega_rans.npy', omega_rans)
np.save('data/Cy_rans', cy)

# load and save rans+ml fields
U_rans_ml = read_field(f'3_pc_Re_tau_550_small_dt/{ml_max_step}/U',mesh_size, n_cells_along_x)
k_rans_ml = read_field(f'3_pc_Re_tau_550_small_dt/{ml_max_step}/k',mesh_size, n_cells_along_x)
omega_rans_ml = read_field(f'3_pc_Re_tau_550_small_dt/{ml_max_step}/omega',mesh_size, n_cells_along_x)
np.save('data/U_rans_ml_small_dt.npy', U_rans_ml)
np.save('data/k_rans_ml_small_dt.npy', k_rans_ml)
np.save('data/omega_rans_ml_small_dt.npy', omega_rans_ml)