import h5py
import numpy as np
import time
import argparse
from pathlib import Path
import sys
from tqdm import tqdm


def make_hdf5_region():
    # load hdf5 file
    f = h5py.File(hdf5_path)
    lats = f['latitude'][:]         # 90 ~ -90  !!! from North to South !!!
    lons = f['longitude'][:]        # 0 ~ 360  !!! from Prime Meridian (London) !!!

    ###################
    # region boundary #
    #   (240, 280)    #
    # 0 N ~ 60 N      #
    # 70 E ~ 140 E    #
    ###################
    ###################
    # image boundary  #
    #   (256, 320)    #
    # 62 N ~ -1.75 S  #
    # 65 E ~ 144.75 E #
    ###################
    # get region lats
    lat_min, lat_max = 0, 60
    lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    lat_min_idx, lat_max_idx = lat_idx[0], lat_idx[-1]
    if(lats[lat_min_idx] != 60) or (lats[lat_max_idx] != 0):
        print("Errors in lats from 90 North ~ -90 South")
    # for better suit for swin v2 downscale
    # 241 --> 256
    # 60N++ ~ 0N--
    lats_region = lats[lat_min_idx-8: lat_max_idx+8]


    # get region lons
    lon_min, lon_max = 70, 140
    lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    lon_min_idx, lon_max_idx = lon_idx[0], lon_idx[-1]
    if(lons[lon_min_idx] != 70) or (lons[lon_max_idx] != 140):
        print("Errors in lons from 0 ~ 360")
    # for better suit for swin v2 downscale
    # 281 --> 320
    # 70N-- ~ 140N++
    lons_region = lons[lon_min_idx-20: lon_max_idx+20]


    # test speed
    # start_time = time.time()
    # data = f['fields'][:, :, 10, 10]
    # end_time = time.time()
    # print('total consuming time: ' + str(int(end_time - start_time)) + ' s')


    # get region data
    time_step_len = f['fields'].shape[0]
    vars_len = f['fields'].shape[1]
    lats_len = lats_region.shape[0]
    lons_len = lons_region.shape[0]
    data_region = np.empty((time_step_len, vars_len, lats_len, lons_len), dtype=f['fields'].dtype)
    with tqdm(total=time_step_len) as pbar:
        for i in range(time_step_len):
            data_region[i] = f['fields'][i, :, lat_min_idx-8: lat_max_idx+8, lon_min_idx-20: lon_max_idx+20]
            if(i+1)%100==0:
                pbar.update(100)
    # get others
    time = f['time'][:]
    channel = f['channel'][:]


    # save hdf5 file
    with h5py.File(hdf5_output, 'w') as file:
        file.create_dataset('fields', data=data_region)
        file.create_dataset('time', data=time)
        file.create_dataset('latitude', data=lats_region)
        file.create_dataset('longitude', data=lons_region)
        file.create_dataset('channel', data=channel)
        file.close()


if __name__ == '__main__':
    # add parameters
    parser = argparse.ArgumentParser(description='save nc variables')
    parser.add_argument('--year', type=str)
    args = parser.parse_args()
    year = args.year

    # make hdf5 file
    start_time = time.time()
    hdf5_path = '/mnt/data5/hdf5/'+year+'.h5'
    hdf5_output = '/mnt/data5/hdf5_region/'+year+'.h5'
    if Path(hdf5_output).exists():
        print(year + ' has existed')
        sys.exit(0)
    make_hdf5_region()
    end_time = time.time()
    print('total consuming time: '+str(int(end_time-start_time)/60)+' minutes')
    print(year+' has done')