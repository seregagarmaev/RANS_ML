import numpy as np
from modules.tbdt import TBDT
from config import *

regressor = 'TBDT'

# READ DATA
X_train, X_test = np.load('data/X_train.npy'), np.load('data/X_test.npy')
TB_train, TB_test = np.load('data/TB_train.npy'), np.load('data/TB_test.npy')
b_train, b_test = np.load('data/b_train.npy'), np.load('data/b_test.npy')


if regressor == 'TBDT':
    # TBDT hyperparameters
    regularization = True
    regularization_lambda = 1e-15
    write_g = False
    splitting_features = 'all'
    min_samples_leaf = 1

    tree_filename = f'data/TBDT/tree'

    tbdt = TBDT(tree_filename=tree_filename, regularization=regularization, splitting_features=splitting_features,
                regularization_lambda=regularization_lambda, optim_split=True, optim_threshold=100,
                min_samples_leaf=min_samples_leaf)

    tree = tbdt.fit(X_train, b_train, TB_train)

    y_predict, g_tree = tbdt.predict(X_test, TB_test, tree)

    print('y_predict', y_predict.shape)
    print('g_tree', g_tree.shape)

    np.save(f'data/b_TBDT.npy', y_predict.T.reshape((-1, 3, 3)))

