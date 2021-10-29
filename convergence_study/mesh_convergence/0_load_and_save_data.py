import numpy as np
from modules.utils_old import read_field


mesh_sizes = [10, 20, 40, 80, 160, 320, 640, 1280]
fields = ['k', 'nut', 'omega', 'p', 'U']
max_step = 20000
cases_path = 'cases'
n_cells_along_x = 2


for n_cells in mesh_sizes:
    path = f'{cases_path}/pcf_{n_cells}/'
    y_centers = read_field(f'{path}/{max_step}/Cy', n_cells, n_cells_along_x)
    saving_path = f'data/{n_cells}_cells_Cy.npy'
    np.save(saving_path, y_centers)

    for field in fields:
        field_array = read_field(f'{path}/{max_step}/{field}', n_cells, n_cells_along_x)
        saving_path = f'data/{n_cells}_cells_{field}.npy'
        np.save(saving_path, field_array)