import numpy as np


def read_field(path, mesh_size, n_cells_along_x, length_row=21):
    with open(path) as file:
        file_lines = file.readlines()
        n_rows = int(file_lines[length_row])
        raw_array = file_lines[length_row+2:length_row+2+n_rows]
        array = [x.replace('(', '').replace(')', '').replace('\n', '') for x in raw_array]
        array = np.loadtxt(array)
        index = [x for x in range(0, mesh_size, n_cells_along_x)] + \
                [x for x in range(2 * mesh_size, 3 * mesh_size, n_cells_along_x)]
        array = array[index]
    return array

def generate_b(data):
    n_points = data.shape[0]
    b = np.zeros((n_points, 3, 3))
    for i in range(n_points):
        uu = data.loc[i, 'uu+']
        vv = data.loc[i, 'vv+']
        ww = data.loc[i, 'ww+']
        uv = data.loc[i, 'uv+']
        k = 0.5 * (uu + vv + ww)
        r = np.array([[uu, uv, 0],
                      [uv, vv, 0],
                      [0, 0, ww]])
        if k != 0:
            b[i, :, :] = r / 2 / k - 1 / 3 * np.eye(3)
    return b

def R_to_b_ij(array):
    n_points = array.shape[0]
    b = np.zeros((n_points, 3, 3))
    for i in range(n_points):
        uu = array[i, 0]
        vv = array[i, 3]
        ww = array[i, 5]
        uv = array[i, 1]
        k = 0.5 * (uu + vv + ww)
        r = np.array([[uu, uv, array[i, 2]],
                      [uv, vv, array[i, 4]],
                      [array[i, 2], array[i, 4], ww]])
        if k != 0:
            b[i, :, :] = r / 2 / k - 1 / 3 * np.eye(3)
    return b


def generate_S_R(grad_U, omega):
    n_cells = grad_U.shape[0]
    S = np.zeros((n_cells, 3, 3))
    R = np.zeros((n_cells, 3, 3))
    for i in range(n_cells):
        S[i, :, :] = 0.5 / omega[i] * (grad_U[i, :, :] + np.transpose(grad_U[i, :, :]))
        R[i, :, :] = 0.5 / omega[i] * (grad_U[i, :, :] - np.transpose(grad_U[i, :, :]))
    return S, R

def calc_invariants(S, R, num_invariants):
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
