import os
import numpy as np
import h5py
from tqdm import tqdm


if __name__ == '__main__':
    # get h5 file path
    # 1979 ~ 2020
    file_paths = [os.path.join('./train', i) for i in sorted(os.listdir('./train'))]


    # get h5 file
    total_sum_x = np.zeros([71, 256, 320], dtype=np.float64)
    # total_sum_x_squared = np.zeros(71, dtype=np.float64)
    total_sum_n = 0
    with tqdm(total=len(file_paths)) as pbar:
        for file in file_paths:
            with h5py.File(file, 'r') as f:
                data = f['fields'][...]
                # get sum
                total_sum_x += data.sum(axis=0)
                # total_sum_x_squared += (data ** 2).sum(axis=(0, 2, 3))
                total_sum_n += data.shape[0]
                pbar.update(1)


    mean = total_sum_x / total_sum_n
    # std = np.sqrt((total_sum_x_squared / total_sum_n) - (mean ** 2))
    mean = mean.reshape(1, 71, 256, 320)
    # std = std.reshape(1, 71, 1, 1)
    # save mean and std path
    np.save('./stats_haotian_region/region_71var_climatology_means.npy', mean)
    # np.save('./stats_haotian_region/region_71var_stds.npy', std)




