import numpy as np
from catboost import CatBoostRegressor

# prepare train and test datasets
cases = [180, 395, 550, 950, 1000, 2000, 5200]
test_cases = [550]

inv_test, tb_test = [], []
inv_train, tb_train = [], []
wbRe_train, wbRe_test = [], []
re_tau_train, re_tau_test = [], []
b_train = []
b_test = []

for i, case in enumerate(cases):
    print(i+1, case)
    inv = np.load(f'data/{i + 1}_inv.npy')
    tb = np.load(f'data/{i + 1}_tb.npy')
    b = np.load(f'data/{i+1}_b.npy')
    wbRe = np.load(f'data/{i+1}_wbRe.npy')
    re_tau = np.load(f'data/{i + 1}_re_tau.npy')
    if case in test_cases:
        inv_test.append(inv)
        tb_test.append(tb)
        b_test.append(b)
        wbRe_test.append(wbRe)
        re_tau_test.append(re_tau)
    else:
        inv_train.append(inv)
        tb_train.append(tb)
        b_train.append(b)
        wbRe_train.append(wbRe)
        re_tau_train.append(re_tau)

inv_train = np.concatenate(inv_train)
tb_train = np.concatenate(tb_train)
b_train = np.concatenate(b_train)
wbRe_train = np.concatenate(wbRe_train)
re_tau_train = np.concatenate(re_tau_train)
inv_test = np.concatenate(inv_test)
tb_test = np.concatenate(tb_test)
b_test = np.concatenate(b_test)
wbRe_test = np.concatenate(wbRe_test)
re_tau_test = np.concatenate(re_tau_test)

print(tb_train.shape)
print(inv_train.shape)
print(b_train.shape)

tb_train = tb_train.reshape((tb_train.shape[0], tb_train.shape[1] * tb_train.shape[2] * tb_train.shape[3]))
tb_test = tb_test.reshape((tb_test.shape[0], tb_test.shape[1] * tb_test.shape[2] * tb_test.shape[3]))


X_train = np.concatenate((inv_train, wbRe_train.reshape((-1, 1)), re_tau_train.reshape((-1, 1))), axis=1)
X_test = np.concatenate((inv_test, wbRe_test.reshape((-1, 1)), re_tau_test.reshape((-1, 1))), axis=1)

# training of gradient boosting algorithm and predicting a b_ij
b_predicted = np.zeros((b_test.shape[0], 3, 3))
for i, j in ((0, 0), (0, 1), (1, 1), (2, 2)):
    model = CatBoostRegressor(iterations=100, depth=8, l2_leaf_reg=3.18, learning_rate=0.1)
    model.fit(X_train, b_train[:, i, j])
    pred = model.predict(X_test)
    b_predicted[:, i, j] = pred

# save predicted b_ij tensor
np.save(f'data/predicted_b_TBGB.npy', b_predicted)
