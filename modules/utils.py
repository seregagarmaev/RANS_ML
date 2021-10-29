import numpy as np


def read_field(field_name, step, path):
    '''This function reads the fields in the OpenFOAM format.'''

    # read file
    file_path = f'{path}/{step}/{field_name}'
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    # determine the file parameters
    for i, line in enumerate(lines):
        if 'internalField' in line:
            field_type = line[line.find('<')+1 : line.find('>')]
            n_cells = int(lines[i+1])
            start_line = i+3
            end_line = start_line + n_cells
            break

    # format field as numpy array
    field = np.loadtxt([line.strip('()') for line in lines[start_line:end_line]])
    if field_type == 'tensor':
        field = field.reshape((-1, 3, 3))
    if field_type == 'symmTensor':
        template = np.zeros((n_cells, 3, 3))
        for i in range(n_cells):
            template[i,:,:] = np.array([
                [field[i, 0], field[i, 1], field[i, 2]],
                [field[i, 1], field[i, 3], field[i, 4]],
                [field[i, 2], field[i, 4], field[i, 5]],
            ])
        field = template

    return field

def dns_to_R(data):
    '''This function creates Rij field using DNS data.'''
    n_points = data.shape[0]
    R = np.zeros((n_points, 3, 3))
    for i in range(n_points):
        uu = data.loc[i, 'uu+']
        vv = data.loc[i, 'vv+']
        ww = data.loc[i, 'ww+']
        uv = data.loc[i, 'uv+']
        r = np.array([[uu, uv, 0],
                      [uv, vv, 0],
                      [0, 0, ww]])
        R[i, :, :] = r
    return R

def generate_S_R(grad_U, omega):
    '''Generate S and R tensors using U gradients and specific dissipation rate'''
    n_cells = grad_U.shape[0]
    S = np.zeros((n_cells, 3, 3))
    R = np.zeros((n_cells, 3, 3))
    for i in range(n_cells):
        S[i, :, :] = 0.5 / omega[i] * (grad_U[i, :, :] + np.transpose(grad_U[i, :, :]))
        R[i, :, :] = 0.5 / omega[i] * (grad_U[i, :, :] - np.transpose(grad_U[i, :, :]))
    return S, R

def calc_invariants(S, R, num_invariants=5):
    '''Generates 5 invariants based on strain and rotation rate tensors.'''
    n_cells = S.shape[0]
    invariants = np.zeros((n_cells, num_invariants))
    for i in range(n_cells):
        invariants[i, 0] = np.trace(np.dot(S[i, :, :], S[i, :, :]))
        invariants[i, 1] = np.trace(np.dot(R[i, :, :], R[i, :, :]))
        invariants[i, 2] = np.trace(np.dot(S[i, :, :], np.dot(S[i, :, :], S[i, :, :])))
        invariants[i, 3] = np.trace(np.dot(R[i, :, :], np.dot(R[i, :, :], S[i, :, :])))
        invariants[i, 4] = np.trace(
            np.dot(np.dot(R[i, :, :], R[i, :, :]), np.dot(S[i, :, :], S[i, :, :])))
    return invariants

def calc_tensor_basis(S, R):
    '''Calculate 10 basis tensors using strain and rotation rate tensors.'''
    n_cells = S.shape[0]
    T = np.zeros((n_cells, 10, 3, 3))
    for i in range(n_cells):
        s = S[i, :, :]
        r = R[i, :, :]

        T[i, 0, :, :] = s
        T[i, 1, :, :] = np.dot(s, r) - np.dot(r, s)
        T[i, 2, :, :] = np.dot(s, s) - 1. / 3. * np.eye(3) * np.trace(np.dot(s, s))
        T[i, 3, :, :] = np.dot(r, r) - 1. / 3. * np.eye(3) * np.trace(np.dot(r, r))
        T[i, 4, :, :] = np.dot(r, np.dot(s, s)) - np.dot(np.dot(s, s), r)
        T[i, 5, :, :] = np.dot(r, np.dot(r, s)) + np.dot(s, np.dot(r, r)) \
                        - 2. / 3. * np.eye(3) * np.trace(np.dot(s, np.dot(r, r)))
        T[i, 6, :, :] = np.dot(np.dot(r, s), np.dot(r, r)) - np.dot(np.dot(r, r), np.dot(s, r))
        T[i, 7, :, :] = np.dot(np.dot(s, r), np.dot(s, s)) - np.dot(np.dot(s, s), np.dot(r, s))
        T[i, 8, :, :] = np.dot(np.dot(r, r), np.dot(s, s)) + np.dot(np.dot(s, s), np.dot(r, r)) \
                        - 2. / 3. * np.eye(3) * np.trace(np.dot(np.dot(s, s), np.dot(r, r)))
        T[i, 9, :, :] = np.dot(np.dot(r, np.dot(s, s)), np.dot(r, r)) \
                        - np.dot(np.dot(r, np.dot(r, s)), np.dot(s, r))
    return T

def R_to_b(array):
    n_points = array.shape[0]
    b = np.zeros((n_points, 3, 3))
    for i in range(n_points):
        k = array[i].trace()
        if k != 0:
            b[i, :, :] = array[i] / 2 / k - 1 / 3 * np.eye(3)
    return b

def wbRe(k_array, cy_array, nu):
    '''Calculate wall distance based Reynolds number.'''
    result = []
    for k, cy in zip(k_array, cy_array):
        wb_Re = np.sqrt(k) * min(cy,2-cy) / 50 / nu
        # wb_Re = min(cy, 2 - cy)
        result.append(wb_Re)
    return np.array(result)