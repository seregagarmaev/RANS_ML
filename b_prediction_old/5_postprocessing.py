import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

# read prepared data
# GB
b_GB = np.load(f'data/predicted_b_TBGB.npy')
b_GB_mirrored = np.flip(b_GB, axis=0)
b_GB_mirrored = b_GB_mirrored * np.array([[1, -1, -1],
                                          [-1, 1, -1],
                                          [-1, -1, 1]])
b_GB_extended = np.concatenate((b_GB, b_GB_mirrored))
# RF
b_RF = np.load(f'data/predicted_b_TBRF.npy')
b_RF_mirrored = np.flip(b_GB, axis=0)
b_RF_mirrored = b_RF_mirrored * np.array([[1, -1, -1],
                                          [-1, 1, -1],
                                          [-1, -1, 1]])
b_RF_extended = np.concatenate((b_RF, b_RF_mirrored))
# NN
# b_NN = np.load(f'data/predicted_b_TBNN.npy')
# b_NN_mirrored = np.flip(b_GB, axis=0)
# b_NN_mirrored = b_NN_mirrored * np.array([[1, -1, -1],
#                                           [-1, 1, -1],
#                                           [-1, -1, 1]])
# b_NN_extended = np.concatenate((b_NN, b_NN_mirrored))


# apply filter
b_GB_filtered = gaussian_filter1d(b_GB_extended, 10, axis=0)
b_RF_filtered = gaussian_filter1d(b_RF_extended, 10, axis=0)
# b_NN_filtered = gaussian_filter1d(b_NN_extended, 10, axis=0)


# cut half of channel
b_GB_filtered = b_GB_filtered[:10000]
b_RF_filtered = b_RF_filtered[:10000]
# b_NN_filtered = b_NN_filtered[:1000]


# saving obtained data
np.save(f'data/predicted_b_TBGB_filtered.npy', b_GB_filtered)
np.save(f'data/predicted_b_TBRF_filtered.npy', b_RF_filtered)
# np.save(f'data/predicted_b_TBNN_filtered.npy', b_NN_filtered)
