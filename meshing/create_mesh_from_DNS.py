# In this script we create mesh description files from DNS raw_data
import numpy as np
import pandas as pd

# change here the path to DNS preprocessed half-channel raw_data
dns_path = '../../OpenFOAM/DNS/4_pc_Re_tau_950/half_channel_data.csv'

# Preparation of y-coordinates array
data = pd.read_csv(dns_path)
print(data)
# there exist 2 column names - 'y/h' and 'y/delta'
y_init = data['y/h'].values
print('Initial y length', len(y_init))
# if loaded raw_data has the middle point put [1:] at the end
# [:] otherwise
y_extension = 2 - np.flip(y_init)[1:]
print('Length of y extension', len(y_extension))
y_final = np.concatenate((y_init, y_extension))
y_final = np.flip(y_final)
ylen = len(y_final)
print('Final y length', ylen)


# Preparation of blockMeshDict
vertices = ''
blocks = ''
inlets = ''
outlets = ''
sides = ''
walls = f'            ({0+(ylen-1)*4} {1+(ylen-1)*4} {3+(ylen-1)*4} {2+(ylen-1)*4})\n            (0 2 3 1)\n'

for y in y_final:
    v1 = f'    (0 {y} 0)\n'
    v2 = f'    (0.1 {y} 0)\n'
    v3 = f'    (0 {y} 0.1)\n'
    v4 = f'    (0.1 {y} 0.1)\n'
    vertices += v1 + v2 + v3 + v4

for n in range(ylen-1):
    p = 4 * n  # period
    block = f'    hex ({4+p} {5+p} {1+p} {0+p} {6+p} {7+p} {3+p} {2+p}) (2 1 2) simpleGrading (1 1 1)\n'
    blocks += block
    inlet = f'            ({4+p} {6+p} {2+p} {0+p})\n'
    inlets += inlet
    outlet = f'            ({7+p} {5+p} {1+p} {3+p})\n'
    outlets += outlet
    s1 = f'            ({4+p} {0+p} {1+p} {5+p})\n'
    s2 = f'            ({6+p} {7+p} {3+p} {2+p})\n'
    sides += s1 + s2

final_dict = f'''/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  2.3.0                                 |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

convertToMeters 1;

vertices
(
{vertices}
);

blocks
(
{blocks}
);

edges
(
);

boundary
(
    walls
    {{
        type            wall;
        faces           
        (
{walls}
        );
    }}
    inlet
    {{
        type cyclic;
        neighbourPatch outlet;
        faces
        (
{inlets}
        );
    }}
    outlet
    {{
        type cyclic;
        neighbourPatch inlet;
        faces
        (
{outlets}
        );
    }}
    sides
    {{
        type            empty;
        faces           
        (
{sides}
        );
    }}

);

mergePatchPairs
(
);

// ************************************************************************* //'''

# Saving of blockMeshDict
with open('blockMeshDict', 'w') as file:
    file.write(final_dict)