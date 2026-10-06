import torch
import numpy as np
import types
import torch.distributed as dist
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import utils.dali_era5_es_helper as esh


def get_data_loader(params, files_pattern, distributed, train):
    # creat dataloader and dataset
    dataloader = DaliDataLoader(params, files_pattern, train)
    dataset = types.SimpleNamespace(img_shape_x=dataloader.img_shape_x, img_shape_y=dataloader.img_shape_y)

    # get train and validate
    if train:
        return dataloader, dataset, None
    else:
        return dataloader, dataset


class DaliDataLoader(object):
    def get_pipeline(self):
        pipeline = Pipeline(batch_size = self.batch_size,
                            num_threads = 32,                             # CPU threads
                            device_id = self.device_index,                # GPU ID
                            py_num_workers = self.num_data_workers,       # callback number (default: 1)
                            py_start_method='spawn',                      # new interpreter (default: fork)
                            seed = self.global_seed,
                            )

        # get input and target
        with pipeline:
            data = fn.external_source(source = esh.ERA5ES(self.location,
                                                          self.train,
                                                          self.batch_size,
                                                          self.dt,
                                                          self.img_size,
                                                          self.n_in_channels,
                                                          self.n_out_channels,
                                                          self.num_shards,     # world size from params.data_num_shards
                                                          self.shard_id,       # world rank form params.data_shard_id
                                                          self.n_future,
                                                          enable_logging = False,
                                                          add_zenith=self.add_zenith,
                                                          seed=self.global_seed,
                                                          ),
                                      num_outputs = 4 if self.add_zenith else 2,  # data length
                                      layout = ["CHW", "CHW"],                    # (c, h, w)
                                      batch = False,
                                      no_copy = True,
                                      parallel = True,
                                      prefetch_queue_depth = self.num_data_workers,
                                      )

            # get inp_zen, tar_zen
            if self.add_zenith:
                inp, tar, izen, tzen = data
            else:
                inp, tar = data

            # upload inp_zen, tar_zen to GPU
            inp = inp.gpu()
            tar = tar.gpu()
            if self.add_zenith:
                izen = izen.gpu()
                tzen = tzen.gpu()

            # normalize inp_zen, tar_zen
            if self.normalize:
                inp = fn.normalize(inp,
                                   device = "gpu",
                                   axis_names = "HW",
                                   batch = False,
                                   mean = self.in_bias,
                                   stddev = self.in_scale)

                tar = fn.normalize(tar,
                                   device = "gpu",
                                   axis_names = "HW",
                                   batch = False,
                                   mean = self.out_bias,
                                   stddev = self.out_scale)

            # output inp, tar, inp_zen, tar_zen
            if self.add_zenith:
                pipeline.set_outputs(inp, tar, izen, tzen)
            else:
                pipeline.set_outputs(inp, tar)

        return pipeline

    def __init__(self, params, location, train, seed = 333):
        # set up seeds (have to be constant)
        self.global_seed = seed                                  # the same on all ranks
        self.local_seed = self.global_seed + dist.get_rank()     # the diffferent seed for every rank

        # load params
        self.num_data_workers = params.num_data_workers
        self.device_index = torch.cuda.current_device()
        self.batch_size = int(params.local_batch_size)
        self.location = location
        self.train = train
        self.dt = params.dt
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)
        self.n_future = params.n_future
        self.img_size = params.img_size
        self.add_zenith = params.add_zenith

        # load mean and std
        self.normalize = True
        means = np.load(params.global_means_path)[0][:self.n_in_channels]
        stds = np.load(params.global_stds_path)[0][:self.n_in_channels]
        self.in_bias = means
        self.in_scale = stds
        means = np.load(params.global_means_path)[0][:self.n_out_channels]
        stds = np.load(params.global_stds_path)[0][:self.n_out_channels]
        self.out_bias = means
        self.out_scale = stds

        # set sharding
        if dist.is_initialized():
            self.num_shards = params.data_num_shards
            self.shard_id = params.data_shard_id
            # print("number of data shards "+str(params['data_num_shards']))
        else:
            self.num_shards = 1
            self.shard_id = 0

        # get img source data
        extsource = esh.ERA5ES(self.location,
                               self.train,
                               self.batch_size,
                               self.dt,
                               self.img_size,
                               self.n_in_channels,
                               self.n_out_channels,
                               self.num_shards,
                               self.shard_id,
                               self.n_future,
                               add_zenith=self.add_zenith,
                               seed=self.global_seed)
        self.num_batches = extsource.num_steps_per_epoch
        self.img_shape_x = extsource.img_shape_x
        self.img_shape_y = extsource.img_shape_y
        del extsource                              # save memory
 
        # create pipeline
        self.pipeline = self.get_pipeline()
        self.pipeline.start_py_workers()
        self.pipeline.build()

        # create iterator
        outnames = ["inp", "tar"]
        if self.add_zenith:
            outnames += ["izen", "tzen"]

        # create iterator
        self.iterator = DALIGenericIterator([self.pipeline],
                                            outnames,
                                            auto_reset=True,
                                            size=-1,
                                            last_batch_policy=LastBatchPolicy.DROP,
                                            prepare_first_batch=True,
                                            )
        
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        #self.iterator.reset()
        for token in self.iterator:
            inp = token[0]["inp"]
            tar = token[0]["tar"]

            if self.add_zenith:
                izen = token[0]["izen"]
                tzen = token[0]["tzen"]
                result = inp, tar, izen, tzen
            else:
                result = inp, tar

            yield result
