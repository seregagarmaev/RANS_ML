import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


################# PREPARE RESIDUALS #########################
fields = ['k', 'nut', 'omega', 'U']
data_path = f'data/'
residuals_dict = {}
start_step = 10000
max_step = 40000

for field in fields:
    print(field)
    field_by_steps = np.load(f'{data_path}{field}.npy')
    if field == 'U':
        field_by_steps = field_by_steps[:,:,0]
    residuals = []
    for n in range(field_by_steps.shape[0] - 1):
        residual = (field_by_steps[n+1] - field_by_steps[n]) / field_by_steps[n+1]
        residual = abs(residual)
        residual = residual[residual <= np.finfo(np.float64).max]
        max_residual = max(residual)
        residuals.append(max_residual)
    residuals_dict[field] = residuals
    print(max(residuals))

################### PLOTTING RESIDUALS #######################
figure = plt.figure(figsize=(10, 5))
for field in fields:
    residuals = residuals_dict[field][::10]
    # sns.lineplot(range(0, max_step-1, 10), residuals, label=field, linestyle = 'dashed')
    plt.plot(range(start_step, max_step-1, 10), residuals, label=field, linestyle = 'dashed')
    plt.yscale('log')
# plt.title('Iterations convergence')
plt.legend()
plt.ylabel('residual')
plt.xlabel('iteration')
plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
plt.savefig(f'plots/iterations_convergence.png', bbox_inches='tight')
