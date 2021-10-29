import numpy as np
from catboost import CatBoostRegressor

# prepare train and test datasets
cases = [180, 395, 550, 950, 1000, 2000, 5200]
test_cases = [550]

inv_test, tb_test = [], []
inv_train, tb_train = [], []
R_train = []
R_test = []

for i, case in enumerate(cases):
    print(i+1, case)
    inv = np.load(f'data/{i + 1}_inv.npy')
    tb = np.load(f'data/{i + 1}_tb.npy')
    R = np.load(f'data/{i+1}_R.npy')
    if case in test_cases:
        inv_test.append(inv)
        tb_test.append(tb)
        R_test.append(R)
    else:
        inv_train.append(inv)
        tb_train.append(tb)
        R_train.append(R)

inv_train = np.concatenate(inv_train)
tb_train = np.concatenate(tb_train)
R_train = np.concatenate(R_train)
inv_test = np.concatenate(inv_test)
tb_test = np.concatenate(tb_test)
R_test = np.concatenate(R_test)

print(tb_train.shape)
print(inv_train.shape)
print(R_train.shape)

tb_train = tb_train.reshape((tb_train.shape[0], tb_train.shape[1] * tb_train.shape[2] * tb_train.shape[3]))
tb_test = tb_test.reshape((tb_test.shape[0], tb_test.shape[1] * tb_test.shape[2] * tb_test.shape[3]))

X_train = np.concatenate((tb_train, inv_train), axis=1)
X_test = np.concatenate((tb_test, inv_test), axis=1)


# training of gradient boosting algorithm and predicting a R_ij
R_predicted = np.zeros((R_test.shape[0], 3, 3))
for i, j in ((0, 0), (0, 1), (1, 1), (2, 2)):
    model = CatBoostRegressor(iterations=100, depth=8, l2_leaf_reg=3.18, learning_rate=0.1)
    model.fit(X_train, R_train[:, i, j])
    pred = model.predict(X_test)
    R_predicted[:, i, j] = pred

# save predicted R_ij tensor
np.save(f'data/predicted_R_TBGB.npy', R_predicted)