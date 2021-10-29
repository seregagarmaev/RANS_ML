import matplotlib.pyplot as plt
import numpy as np


################# PREPARE RESIDUALS #########################
fields = ['k', 'nut', 'omega', 'U']
data_path = f'data/'
residuals_dict = {}


steps = [n for n in range(0, 100, 1)]
steps += [n for n in range(100, 1000, 10)]
steps += [n for n in range(1000, 10000, 100)]
steps += [n for n in range(10000, 20000, 1000)]


for field in fields:
    print(field)
    field_by_steps = np.load(f'{data_path}{field}.npy')
    if field == 'U':
        field_by_steps = field_by_steps[:,:,0]
    residuals = []
    for i in range(len(steps)-1):
        residual = (field_by_steps[steps[i+1]] - field_by_steps[steps[i]]) / \
                   (np.amax(field_by_steps[steps[i+1]]) - np.amin(field_by_steps[steps[i+1]]))
        max_residual = np.amax(residual)
        residuals.append(max_residual)
    residuals_dict[field] = residuals


################### PLOTTING RESIDUALS #######################
figure = plt.figure(figsize=(10, 5))
for field in fields:
    residuals = residuals_dict[field]
    plt.plot(steps[:-1], residuals, label=field, linestyle = 'dashed')
    plt.yscale('log')
plt.title('Iterations convergence study. Step 4(b)')
plt.legend()
plt.ylabel('residual')
plt.xlabel('iteration')
plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
plt.savefig(f'plots/4b.png', bbox_inches='tight')