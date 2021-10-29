import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae
from scipy.interpolate import interp1d


b_true = np.load(f'data/2_b.npy')

b_GB = np.load(f'data/predicted_b_TBGB_filtered.npy')
b_RF = np.load(f'data/predicted_b_TBRF_filtered.npy')
b_NN = np.load(f'data/predicted_b_TBNN_filtered.npy')
b_rans = np.load(f'data/3_b_RANS.npy')

cy_rans = np.load(f'raw_data/3_rans_Cy.npy')
ls = np.load(f'data/3_space.npy')

b_interpolant = interp1d(cy_rans, b_rans, axis=0, kind='cubic')
b_rans = b_interpolant(ls)


for i, j in ((0, 0), (0, 1), (1, 1), (2, 2)):
    print('b', i + 1, j + 1)
    print('GB MSE:', mse(b_true[:, i, j], b_GB[:, i, j]))
    print('GB MAE:', mae(b_true[:, i, j], b_GB[:, i, j]))
    print('GB MAPE:', mape(b_true[:, i, j], b_GB[:, i, j]))
    print()

    print('RF MSE:', mse(b_true[:, i, j], b_RF[:, i, j]))
    print('RF MAE:', mae(b_true[:, i, j], b_RF[:, i, j]))
    print('RF MAPE:', mape(b_true[:, i, j], b_RF[:, i, j]))
    print()

    print('NN MSE:', mse(b_true[:, i, j], b_NN[:, i, j]))
    print('NN MAE:', mae(b_true[:, i, j], b_NN[:, i, j]))
    print('NN MAPE:', mape(b_true[:, i, j], b_NN[:, i, j]))
    print()

    print('RANS MSE:', mse(b_true[:, i, j], b_rans[:, i, j]))
    print('RANS MAE:', mae(b_true[:, i, j], b_rans[:, i, j]))
    print('RANS MAPE:', mape(b_true[:, i, j], b_rans[:, i, j]))
    print()

    print('=' * 50)