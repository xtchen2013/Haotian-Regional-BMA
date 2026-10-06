import h5py
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # load hdf5 file
    f = h5py.File("/mnt/data5/hdf5/1979.h5")
    lats = f['latitude'][:]         # 90 ~ -90  !!! from North to South !!!
    lons = f['longitude'][:]        # 0 ~ 360  !!! from Prime Meridian (London) !!!
    # get lsm
    lsm = np.load("./invariants_haotian/lsm.npy")
    # get orog
    orog = h5py.File("./invariants_haotian/orography.h5")['orog']


    ###################
    # region boundary #
    # 0 N ~ 60 N      #
    # 70 E ~ 140 E    #
    ###################
    # get region lats
    lat_min, lat_max = 0, 60
    lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lat_min_idx, lat_max_idx = lat_idx[0], lat_idx[-1]
    if(lats[lat_min_idx] != 60) or (lats[lat_max_idx] != 0):
        print("Errors in lats from 90 North ~ -90 South")


    # get region lons
    lon_min, lon_max = 70, 140
    lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    lon_min_idx, lon_max_idx = lon_idx[0], lon_idx[-1]
    if(lons[lon_min_idx] != 70) or (lons[lon_max_idx] != 140):
        print("Errors in lons from 0 ~ 360")


    # get region data
    # for better suit for swin v2 downscale
    # 241 --> 256
    # 60N++ ~ 0N--
    # 281 --> 320
    # 70N-- ~ 140N++
    lsm = lsm[lat_min_idx-8: lat_max_idx+8, lon_min_idx-20: lon_max_idx+20]
    # plt.imshow(lsm)
    # plt.show()
    np.save('./invariants_haotian_region/lsm.npy', lsm)
    orog = orog[lat_min_idx-8: lat_max_idx+8, lon_min_idx-20: lon_max_idx+20]
    # plt.imshow(orog)
    # plt.show()
    with h5py.File("./invariants_haotian_region/orography.h5", 'w') as f:
        f.create_dataset('orog', data=orog)


