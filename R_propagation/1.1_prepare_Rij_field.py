import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


# constants
u_tau_dns = 5.43496e-02

# read prepared data
R_RF = np.load(f'../R_prediction/data/predicted_R_TBRF_filtered.npy')
R_RF_mirrored = np.flip(R_RF, axis=0)
R_RF_mirrored = R_RF_mirrored * np.array([[1, -1, -1],
                                          [-1, 1, -1],
                                          [-1, -1, 1]])
R_RF_extended = np.concatenate((R_RF, R_RF_mirrored))
cell_centers = np.load(f'../R_prediction/data/3_space.npy')
cell_centers = np.concatenate((cell_centers, 2 - np.flip(cell_centers)))
rans_cell_centers = np.load(f'../R_prediction/raw_data/3_rans_Cy.npy')
# there is a small deviation - we eliminate this
cell_centers[-1] = rans_cell_centers[-1]

# apply filter
# b_RF_filtered = gaussian_filter1d(b_RF_extended, 10, axis=0)

# interpolate on rans cell centers
R_interpolant = interp1d(cell_centers, R_RF_extended, axis=0, kind='cubic')
R_RF_filtered_intd = R_interpolant(rans_cell_centers)

# undimesionalize
R_RF_filtered_intd = R_RF_filtered_intd * u_tau_dns * u_tau_dns

# prepare and save R_ij field in OpenFOAM format
Rij_field = ''
for i in range(2):
    for j in range(int(R_RF_filtered_intd.shape[0]/2)):
        x = R_RF_filtered_intd[j]
        Rij_field += f'({x[0, 0]} {x[0, 1]} {x[0, 2]} {x[1, 1]} {x[1, 2]} {x[2, 2]})\n'
        Rij_field += f'({x[0, 0]} {x[0, 1]} {x[0, 2]} {x[1, 1]} {x[1, 2]} {x[2, 2]})\n'

for i in range(2):
    for j in range(int(R_RF_filtered_intd.shape[0]/2), int(R_RF_filtered_intd.shape[0])):
        x = R_RF_filtered_intd[j]
        Rij_field += f'({x[0, 0]} {x[0, 1]} {x[0, 2]} {x[1, 1]} {x[1, 2]} {x[2, 2]})\n'
        Rij_field += f'({x[0, 0]} {x[0, 1]} {x[0, 2]} {x[1, 1]} {x[1, 2]} {x[2, 2]})\n'

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
    object      R;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];


internalField   nonuniform List<symmTensor> 
{R_RF_filtered_intd.shape[0]*4}
(
{Rij_field}
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

# Saving R_ij as a text file
with open('R', 'w') as file:
    file.write(file_template)