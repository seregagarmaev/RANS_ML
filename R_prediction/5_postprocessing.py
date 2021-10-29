import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# read prepared data
# GB
R_GB = np.load(f'data/predicted_R_TBGB.npy')
R_GB_mirrored = np.flip(R_GB, axis=0)
R_GB_mirrored = R_GB_mirrored * np.array([[1, -1, -1],
                                          [-1, 1, -1],
                                          [-1, -1, 1]])
R_GB_extended = np.concatenate((R_GB, R_GB_mirrored))
# RF
R_RF = np.load(f'data/predicted_R_TBRF.npy')
R_RF_mirrored = np.flip(R_GB, axis=0)
R_RF_mirrored = R_RF_mirrored * np.array([[1, -1, -1],
                                          [-1, 1, -1],
                                          [-1, -1, 1]])
R_RF_extended = np.concatenate((R_RF, R_RF_mirrored))
# NN
# b_NN = np.load(f'data/predicted_b_TBNN.npy')
# b_NN_mirrored = np.flip(b_GB, axis=0)
# b_NN_mirrored = b_NN_mirrored * np.array([[1, -1, -1],
#                                           [-1, 1, -1],
#                                           [-1, -1, 1]])
# b_NN_extended = np.concatenate((b_NN, b_NN_mirrored))


# apply filter
R_GB_filtered = gaussian_filter1d(R_GB_extended, 10, axis=0)
R_RF_filtered = gaussian_filter1d(R_RF_extended, 10, axis=0)
# b_NN_filtered = gaussian_filter1d(b_NN_extended, 10, axis=0)


# cut half of channel
R_GB_filtered = R_GB_filtered[:1000]
R_RF_filtered = R_RF_filtered[:1000]
# b_NN_filtered = b_NN_filtered[:1000]


# saving obtained data
np.save(f'data/predicted_R_TBGB_filtered.npy', R_GB_filtered)
np.save(f'data/predicted_R_TBRF_filtered.npy', R_RF_filtered)
# np.save(f'data/predicted_b_TBNN_filtered.npy', b_NN_filtered)
