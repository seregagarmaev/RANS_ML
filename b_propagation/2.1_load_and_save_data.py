import numpy as np
from modules.utils import read_field

dir_path = '3_pc_Re_tau_550'
fields = ['k', 'nut', 'omega', 'p', 'U']
start_step = 20000
max_step = 40000
mesh_size = 640
n_cells_along_x = 2

for field in fields:
    field_by_steps = []
    for step in range(start_step, max_step+1):
        if step % 100 == 0:
            print(step)
        path = f'{dir_path}/{step}/{field}'
        field_array = read_field(path, mesh_size, n_cells_along_x)
        field_by_steps.append(field_array)
    field_by_steps = np.array(field_by_steps)
    saving_path = f'data/{field}.npy'
    np.save(saving_path, field_by_steps)
    print(f'Saved {field} field to {saving_path}')