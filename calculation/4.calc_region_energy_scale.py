import os
import numpy as np
import h5py
from tqdm import tqdm


if __name__ == '__main__':
    # get h5 file path
    file_paths = [os.path.join('./train', i) for i in sorted(os.listdir('./train'))]


    # get idx
    u_idx = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    v_idx = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    t_idx = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
    q_idx = [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
    z_idx = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]

    # get h5 file
    total_sum_ke = np.zeros(13, dtype=np.float64)
    total_sum_pe = np.zeros(13, dtype=np.float64)
    total_sum_ie = np.zeros(13, dtype=np.float64)
    total_sum_le = np.zeros(13, dtype=np.float64)
    total_sum_n = np.zeros(13, dtype=np.int64)
    with tqdm(total=len(file_paths)) as pbar:
        for file in file_paths:
            with h5py.File(file, 'r') as f:
                data = f['fields'][...]
                # get vars
                u = data[:, u_idx, :, :]
                v = data[:, v_idx, :, :]
                t = data[:, t_idx, :, :]
                q = data[:, q_idx, :, :]
                z = data[:, z_idx, :, :]
                # calculate energy
                ke = 0.5 * (u ** 2 + v ** 2)
                pe = 9.80665 * z
                ie = 1004 * t
                le = (2.501e6 - 2369.5 * (t - 273.15)) * q
                # get sum
                for i in range(13):
                    total_sum_ke[i] += ke[:, i, :, :].sum()
                    total_sum_pe[i] += pe[:, i, :, :].sum()
                    total_sum_ie[i] += ie[:, i, :, :].sum()
                    total_sum_le[i] += le[:, i, :, :].sum()
                    total_sum_n[i] += data[:, 0, :, :].size
                pbar.update(1)

    ke_scale = total_sum_ke / total_sum_n
    pe_scale = total_sum_pe / total_sum_n
    ie_scale = total_sum_ie / total_sum_n
    le_scale = total_sum_le / total_sum_n
    # save mean and std path
    np.save('./stats_haotian_region/region_kinetic_energy_scale.npy', ke_scale)
    np.save('./stats_haotian_region/region_potential_energy_scale.npy', pe_scale)
    np.save('./stats_haotian_region/region_internal_energy_scale.npy', ie_scale)
    np.save('./stats_haotian_region/region_latent_energy_scale.npy', le_scale)






