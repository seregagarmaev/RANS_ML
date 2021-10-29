import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns


mesh_sizes = [10, 20, 40, 80, 160, 320, 640]  # , 1280]
field_names = ['k', 'nut', 'omega', 'U']
max_step = 6000
cell_centers = {}
n_cells_along_x = 2

for n_cells in mesh_sizes:
    cell_centers[n_cells] = np.load(f'data/{n_cells}_cells_Cy.npy')


plt.figure(figsize=(10, 5))

for field_name in field_names:
    print(field_name)
    residuals = []
    for i in range(len(mesh_sizes)-1):
        print(mesh_sizes[i])
        # reading fields
        # left_index = [x for x in range(0, mesh_sizes[i], n_cells_along_x)] + \
        #              [x for x in range(2 * mesh_sizes[i], 3 * mesh_sizes[i], n_cells_along_x)]
        # right_index = [x for x in range(0, mesh_sizes[i+1], n_cells_along_x)] + \
        #              [x for x in range(2 * mesh_sizes[i+1], 3 * mesh_sizes[i+1], n_cells_along_x)]
        left_cell_centers = cell_centers[mesh_sizes[i]]  # [left_index]
        right_cell_centers = cell_centers[mesh_sizes[i+1]]  # [right_index]
        left_field = np.load(f'data/{mesh_sizes[i]}_cells_{field_name}.npy')  # [left_index]
        right_field = np.load(f'data/{mesh_sizes[i+1]}_cells_{field_name}.npy')  # [right_index]
        if field_name == 'U':
            left_field = left_field[:, 0]
            right_field = right_field[:, 0]

        if field_name == 'omega':
            left_field = 1 / left_field
            right_field = 1 / right_field

        # making interpolation
        right_interpolant = interp1d(right_cell_centers, right_field, kind='cubic')
        right_field_interpolated = right_interpolant(left_cell_centers)
        residual = np.abs(np.amax(right_field_interpolated - left_field)) / \
                   (np.amax(right_field_interpolated) - np.amin(right_field_interpolated))
        residuals.append(residual)
    if field_name == 'omega':
        label = '1/omega'
    else:
        label = field_name
    sns.lineplot(mesh_sizes[1:], residuals, label=label)
plt.ylabel('residuals')
plt.xlabel('number of cells')
plt.yscale('log')
plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
# plt.title(f'{field_name} field residuals for mesh with simple grading')
plt.savefig(f'plots/4g.png', bbox_inches='tight')
