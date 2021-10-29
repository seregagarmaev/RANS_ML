import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, Trials
from hyperopt import hp

# prepare train and test datasets
cases = [180, 395, 550, 950, 1000, 2000, 5200]
test_cases = [550]

inv_test, tb_test = [], []
inv_train, tb_train = [], []
b_train = []
b_test = []

for i, case in enumerate(cases):
    print(i+1, case)
    inv = np.load(f'data/{i + 1}_inv.npy')
    tb = np.load(f'data/{i + 1}_tb.npy')
    b = np.load(f'data/{i+1}_b.npy')
    if case in test_cases:
        inv_test.append(inv)
        tb_test.append(tb)
        b_test.append(b)
    else:
        inv_train.append(inv)
        tb_train.append(tb)
        b_train.append(b)

inv_train = np.concatenate(inv_train)
tb_train = np.concatenate(tb_train)
b_train = np.concatenate(b_train)
inv_test = np.concatenate(inv_test)
tb_test = np.concatenate(tb_test)
b_test = np.concatenate(b_test)

tb_train = tb_train.reshape((tb_train.shape[0], tb_train.shape[1] * tb_train.shape[2] * tb_train.shape[3]))
tb_test = tb_test.reshape((tb_test.shape[0], tb_test.shape[1] * tb_test.shape[2] * tb_test.shape[3]))

X_train = np.concatenate((tb_train, inv_train), axis=1)
X_test = np.concatenate((tb_test, inv_test), axis=1)

def get_rf_params(space):
    params = dict()
    params['n_estimators'] = int(space['n_estimators'])
    params['max_depth'] = int(space['max_depth'])
    params['min_samples_leaf'] = int(space['min_samples_leaf'])
    params['min_samples_split'] = int(space['min_samples_split'])
    return params

def objective(space):
    params = get_rf_params(space)
    model = RandomForestRegressor(n_estimators=params['n_estimators'],
                                  max_depth=params['max_depth'],
                                  min_samples_leaf=params['min_samples_leaf'],
                                  min_samples_split=params['min_samples_split'])
    mse = 0
    for i, j in ((0, 0), (0, 1), (1, 1), (2, 2)):
        model.fit(X_train, b_train[:, i, j])
        y_pred = model.predict(X_test)
        mse += mean_squared_error(b_test[:, i, j], y_pred)
    return mse


space = {
    'n_estimators':hp.uniform('n_estimators',50,500),
    'max_depth':hp.uniform('max_depth',2,20),
    'min_samples_leaf':hp.uniform('min_samples_leaf',1,5),
    'min_samples_split':hp.uniform('min_samples_split',2,6),
}


opt_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=Trials())
print(opt_params)
