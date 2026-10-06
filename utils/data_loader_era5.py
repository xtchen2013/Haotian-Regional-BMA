import datetime
import glob
import h5py
import logging
from utils.zenith_angle import cos_zenith_angle
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple


def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed % (2 ** 32 - 1))


def is_leap_year(yr):
    return yr % 4 == 0


def get_data_loader(params, files_pattern, distributed, train):
    # get dataset
    dataset = GetDataset(params, files_pattern, train)

    # get distributed sampler
    if distributed:
        sampler = DistributedSampler(dataset,
                                     shuffle=train,
                                     num_replicas=params.data_num_shards,
                                     rank=params.data_shard_id,
                                     )
    else:
        sampler = None

    # get dataloader
    dataloader = DataLoader(dataset,
                            batch_size=int(params.batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=(sampler is None),
                            sampler=sampler,
                            worker_init_fn=worker_init,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available(),
                            )

    # return
    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


class GetDataset(Dataset):
    def __init__(self, params, location, train):
        # get initial attributes
        self.params = params                                                                # swin yaml
        self.location = location                                                            # files path
        self.train = train                                                                  # bool
        self.dt = params.dt                                                                 # time step
        self.in_channels = params.in_channels                                               # input channels
        self.out_channels = params.out_channels                                             # output channels
        self.n_in_channels = params.n_in_channels                                           # number of input channels
        self.n_out_channels = params.n_out_channels                                         # number of output channels
        self.n_future = params.n_future                                                     # for many future time steps
        self.normalize = True                                                               # calculate normalization
        self.means = np.load(params.global_means_path)[0, self.in_channels]                 # (71, 1, 1)
        self.stds = np.load(params.global_stds_path)[0, self.in_channels]                   # (71, 1, 1)

        # get files stats
        self._get_files_stats()

        # calculate static variables
        if self.params.add_zenith:
            # additional static fields needed for coszen
            longitude = np.arange(0, 360, 0.25)                                             # 1440: 0 ~ 359.75
            latitude = np.arange(-90, 90.25, 0.25)[::-1]                                    # 721: 90 ~ -90
            self.lon_grid_local, self.lat_grid_local = np.meshgrid(longitude, latitude)

    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.years = [int(os.path.splitext(os.path.basename(x))[0][:4]) for x in self.files_paths]
        self.n_years = len(self.files_paths)

        # do not use leap year unless they are all leap years
        stats_idx = 0
        while is_leap_year(self.years[stats_idx]):
            stats_idx += 1
            if stats_idx >= self.n_years:
                stats_idx = 0
                break

        # check h5 file latitude and longitude
        # print(self.files_paths[stats_idx])
        with h5py.File(self.files_paths[stats_idx], 'r') as _f:
            logging.info("Getting file stats from {}".format(self.files_paths[stats_idx]))
            self.n_samples_per_year = _f['fields'].shape[0]                    # number of samples (365*24/6)
            self.img_shape_x = self.params.img_size[0]                                    # latitude: 720
            self.img_shape_y = self.params.img_size[1]                                    # longitude: 1440
            assert (self.img_shape_x <= _f['latitude'].shape[0] and self.img_shape_y <= _f['longitude'].shape[0]), 'image shapes are greater than dataset image shapes'

        # logging samples
        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))


    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file['fields']


    def __len__(self):
        return self.n_samples_total


    def _normalize(self, img):
        if self.normalize:
            means = self.means
            stds = self.stds
            if len(img.shape) > 3:
                means = np.expand_dims(means, 0)
                stds = np.expand_dims(stds, 0)
            img -= means
            img /= stds
        return torch.as_tensor(img)


    def _compute_zenith_angle(self, local_idx: int, year_idx: int, time_step_hours: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the cosine of the zenith angle for specific time points.

        Parameters:
            -> local_idx (int): Index for the current local time point.
            -> year_idx (int): Index for the year in the years array.
            -> time_step_hours (int, optional): Time step size in hours. Default is 6.
        Returns:
            -> Tuple[torch.Tensor, torch.Tensor]: Tensors for input and target cosine zenith angles.
        """

        if not 0 <= year_idx < len(self.years):
            raise ValueError("year_idx is out of bounds.")
        year = self.years[year_idx]

        # reference datetime for the start of the year
        jan_01_epoch = datetime.datetime(year, 1, 1, 0, 0, 0)

        # helper function to calculate cosine zenith angles
        def calculate_cos_zenith(start_idx: int, end_idx: int) -> np.ndarray:
            # to store cosine zenith angles
            cos_zenith = []
            for idx in range(start_idx, end_idx, self.dt):
                hours_since_jan_01 = idx * time_step_hours
                model_time = jan_01_epoch + datetime.timedelta(hours=hours_since_jan_01)
                # calculate and append the cosine of zenith angle for this time
                cos_zenith.append(cos_zenith_angle(model_time, self.lon_grid_local, self.lat_grid_local).astype(np.float32))

            # stack the angles into a multidimensional array
            return np.stack(cos_zenith, axis=0)

        # calculate the cosine zenith angles for the input and target time points
        cos_zenith_inp = calculate_cos_zenith(local_idx, local_idx + 1)
        cos_zenith_tar = calculate_cos_zenith(local_idx + self.dt, local_idx + self.dt * (self.n_future + 1) + 1)
        # return the input and target angles as PyTorch tensors
        return torch.as_tensor(cos_zenith_inp), torch.as_tensor(cos_zenith_tar)

    def __getitem__(self, global_idx):
        # get sample in that year
        year_idx = int(global_idx / self.n_samples_per_year)                                 # which year
        local_idx = int(global_idx % self.n_samples_per_year)                                # which sample in that year


        # open image file
        if self.files[year_idx] is None:
            self._open_file(year_idx)


        # boundary conditions to ensure we don't pull data that is not in a specific year
        step = self.dt                                                                       # time step
        local_idx = local_idx % (self.n_samples_per_year - step * (self.n_future + 1))
        if local_idx < step:
            local_idx += step


        # pre-process and get the image fields
        # inp_field -> (71, 720, 1440)
        # tar_field -> (step, 71, 720, 1440)
        inp_field = self.files[year_idx][local_idx, self.in_channels, 0:self.img_shape_x, 0:self.img_shape_y]
        tar_field = self.files[year_idx][(local_idx+step):(local_idx+(self.n_future+1)*step+1):step,
                    self.out_channels, 0:self.img_shape_x, 0:self.img_shape_y]


        # normalize images if needed
        inp, tar = self._normalize(inp_field), self._normalize(tar_field)


        # flatten time indices
        # inp -> (71, 720, 1440)
        # tar -> (71*(step+1), 720, 1440)
        tar = tar.reshape((self.n_out_channels * (self.n_future + 1), self.img_shape_x, self.img_shape_y))
        if self.params.add_zenith:
            zen_inp, zen_tar = self._compute_zenith_angle(local_idx, year_idx)  # compute the zenith angles for the input
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            #         Very Important           #
            #       adjust to region           #
            #         --> (256, 320)           #
            #       [112: 368, 260, 580]       #
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
            crop_offset = (120 - 8, 280 - 20)
            crop_shape = (self.params.img_size[0], self.params.img_size[1])
            # zen_inp = zen_inp[:, :self.img_shape_x]                             # adjust to match input dimensions
            # zen_tar = zen_tar[:, :self.img_shape_x]                             # adjust to match input dimensions
            zen_inp = zen_inp[:, crop_offset[0]:crop_offset[0]+crop_shape[0], crop_offset[1]:crop_offset[1]+crop_shape[1]]
            zen_tar = zen_tar[:, crop_offset[0]:crop_offset[0]+crop_shape[0], crop_offset[1]:crop_offset[1]+crop_shape[1]]
            # zen_inp = zen_inp[:, 112:368, 260:580]
            # zen_tar = zen_tar[:, 112:368, 260:580]
            result = inp, tar, zen_inp, zen_tar
        else:
            result = inp, tar
        # print(len(result))
        return result


