import numpy as np
import pandas as pd
from modules.utils_old import read_field

dns_path = 'DNS_cases'
dns_columns = ['y/delta', 'uu+', 'vv+', 'ww+', 'uv+']
rans_path = 'RANS_cases'
field_names = ['Cy', 'grad(U)', 'omega', 'turbulenceProperties:R']
max_step = 10000
mesh_size = 640
n_cells_along_x = 2

re_t_list = [180, 395, 550, 950, 1000, 2000, 5200]

# load and save DNS and RANS raw_data
for i, re_t in enumerate(re_t_list):
    dns_data = pd.read_csv(f'{dns_path}/{i+1}_pc_Re_tau_{re_t}/half_channel_data.csv')[dns_columns]
    dns_data.to_feather(f'raw_data/{i+1}_dns.feather')
    for field_name in field_names:
        rans_field = read_field(f'{rans_path}/{i+1}_pc_Re_tau_{re_t}/{max_step}/{field_name}',
                                mesh_size, n_cells_along_x)
        np.save(f'raw_data/{i + 1}_rans_{field_name}.npy', rans_field)