import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


# read prepared data
b_DNS = np.load(f'../b_prediction_old/data/3_b.npy')
b_DNS_mirrored = np.flip(b_DNS, axis=0)
b_DNS_mirrored = b_DNS_mirrored * np.array([[1, -1, -1],
                                          [-1, 1, -1],
                                          [-1, -1, 1]])
b_DNS_extended = np.concatenate((b_DNS, b_DNS_mirrored))
cell_centers = np.load(f'../b_prediction_old/data/3_space.npy')
cell_centers = np.concatenate((cell_centers, 2 - np.flip(cell_centers)))
rans_cell_centers = np.load(f'../b_prediction_old/raw_data/3_rans_Cy.npy')
# there is a small deviation - we eliminate this
cell_centers[-1] = rans_cell_centers[-1]

# apply filter
# b_RF_filtered = gaussian_filter1d(b_RF_extended, 10, axis=0)

# interpolate on rans cell centers
b_interpolant = interp1d(cell_centers, b_DNS_extended, axis=0, kind='cubic')
b_DNS_filtered_intd = b_interpolant(rans_cell_centers)

# prepare and save b_ij field in OpenFOAM format
b_ij_field = ''
for i in range(2):
    for j in range(int(b_DNS_filtered_intd.shape[0]/2)):
        x = b_DNS_filtered_intd[j]
        b_ij_field += f'({x[0, 0]} {x[0, 1]} {x[0, 2]} {x[1, 1]} {x[1, 2]} {x[2, 2]})\n'
        b_ij_field += f'({x[0, 0]} {x[0, 1]} {x[0, 2]} {x[1, 1]} {x[1, 2]} {x[2, 2]})\n'

for i in range(2):
    for j in range(int(b_DNS_filtered_intd.shape[0]/2), int(b_DNS_filtered_intd.shape[0])):
        x = b_DNS_filtered_intd[j]
        b_ij_field += f'({x[0, 0]} {x[0, 1]} {x[0, 2]} {x[1, 1]} {x[1, 2]} {x[2, 2]})\n'
        b_ij_field += f'({x[0, 0]} {x[0, 1]} {x[0, 2]} {x[1, 1]} {x[1, 2]} {x[2, 2]})\n'

file_template = f'''/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2012                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volSymmTensorField;
    location    "0";
    object      bij;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];


internalField   nonuniform List<symmTensor> 
{b_DNS_filtered_intd.shape[0]*4}
(
{b_ij_field}
)
;
boundaryField
{{
    walls
    {{
        type            fixedValue;
        value           uniform (0 0 0 0 0 0);
    }}
    inlet
    {{
        type            cyclic;
    }}
    outlet
    {{
        type            cyclic;
    }}
    sides
    {{
        type            empty;
    }}
}}
// ************************************************************************* //'''

# Saving b_ij as a text file
with open('bij', 'w') as file:
    file.write(file_template)