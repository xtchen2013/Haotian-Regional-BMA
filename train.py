import os
import time
import numpy as np
import argparse
import torch
import wandb
# import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Callable, Any
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from apex import optimizers
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from utils import get_data_loader_distributed
from utils.weighted_acc_rmse import weighted_rmse_torch
# from utils.img_utils import vis
from utils.preprocess_utils import PreProcessor
from utils.losses import LossHandler
from networks.helpers import get_model
import datetime
import warnings
from timm.scheduler import CosineLRScheduler
warnings.filterwarnings('ignore')


# set offline mode
os.environ["WANDB_API_KEY"] = '0f9a6cc1c83cf5150c25bd5f8e1ca85a51ab2c97'
os.environ["WANDB_MODE"] = "offline"

# set GPUs and CPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"         # v100 16G
# os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4, 5, 6"                     # A800 80G
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"                     # v100 32G
os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"      # for fine-tuning
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"



def ckpt_identity(layer: Callable, *args: Any, **kwargs: Any) -> Any:
    """Identity function for when activation checkpointing is not needed"""
    return layer(*args)


def set_seed(params, world_size):
    seed = params.seed
    if seed is None:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if world_size > 0:
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, params, args):
        self.preprocessor = None
        self.sweep_id = args.sweep_id                                    # sweep.yaml in wandb
        self.root_dir = params['exp_dir']                                # experiment dir
        self.config = args.config                                        # swin.yaml
        params['enable_amp'] = args.enable_amp                           # automatic mixed precision (True or False)

        # one GPU
        self.world_size = 1                                              # number of GPUs
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
        self.local_rank = 0                                              # local rank in a node
        self.world_rank = 0                                              # world rank

        # multi GPUs
        if self.world_size > 1:
            # init_method: PyTorch NGC 23.07 v0
            dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=3600))
            self.world_rank = dist.get_rank()                            # world rank of GPU ID
            self.local_rank = int(os.environ["LOCAL_RANK"])              # local rank of GPU ID

        # set device
        torch.cuda.set_device(self.local_rank)
        torch.backends.cudnn.benchmark = True                            # look for the optimal set of algorithms

        # logging to screen and wandb
        self.log_to_screen = params.log_to_screen and self.world_rank == 0
        self.log_to_wandb = params.log_to_wandb and self.world_rank == 0

        # get device and params
        self.device = torch.cuda.current_device()
        self.params = params
        self.params['name'] = args.config + '_' + str(args.run_num)
        self.params['group'] = args.config

        # for dali data loader, set the actual number of data shards and id
        self.params['data_num_shards'] = self.world_size          # number of GPU
        self.params['data_shard_id'] = self.world_rank            # id of GPU
        self.config = args.config
        self.run_num = args.run_num

        # activate checkpoint for saving memory
        self.ckpt_fn = torch.utils.checkpoint.checkpoint if hasattr(params, 'activation_ckpt') and params.activation_ckpt else ckpt_identity

        # physics loss
        self.physics = params.physics



    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


    ################################
    # build DDP and start training #
    ################################
    def build_and_launch(self):
        # load params
        self.params['in_channels'] = np.array(self.params['in_channels'])
        self.params['out_channels'] = np.array(self.params['out_channels'])
        self.params['n_in_channels'] = len(self.params['in_channels'])
        self.params['n_out_channels'] = len(self.params['out_channels'])

        # calculate by latitude and longitude
        if self.params.add_zenith:
            self.params.n_in_channels += 1

        # lsm.npy
        if self.params.add_landmask:
            self.params.n_in_channels += 2

        # org.h5
        if self.params.add_orography:
            self.params.n_in_channels += 1

        # initialize weights and bias (wandb)
        if self.sweep_id:
            jid = os.environ['SLURM_JOBID']                              # so different sweeps dont resume
            exp_dir = os.path.join(*[self.root_dir, 'sweeps', self.sweep_id, self.config, jid])
        else:
            exp_dir = os.path.join(*[self.root_dir, self.config, self.run_num])

        # master make ckpt and wandb
        if self.world_rank == 0:
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir)
                os.makedirs(os.path.join(exp_dir, 'training_checkpoints/'))
                os.makedirs(os.path.join(exp_dir, 'wandb/'))

        # ckpt path
        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        self.params['checkpoint_path'] = os.path.join(exp_dir, 'training_checkpoints/ckpt.tar')
        self.params['best_checkpoint_path'] = os.path.join(exp_dir, 'training_checkpoints/best_ckpt.tar')
        self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False

        # logging wandb
        if self.log_to_wandb:
            if self.sweep_id:
                wandb.init(dir=os.path.join(exp_dir, "wandb"))
                hpo_config = wandb.config
                self.params.update_params(hpo_config)
                logging.info('HPO sweep %s, trial params:' % self.sweep_id)
                logging.info(self.params.log())
            else:
                wandb.init(dir=os.path.join(exp_dir, "wandb"),
                           config=self.params.params,
                           name=self.params.name,
                           group=self.params.group,
                           project=self.params.project,
                           entity=self.params.entity,
                           resume=self.params.resuming,
                           )
                logging.info(self.params.log())


        # broadcast the params to all ranks since the sweep agent has changed it
        if self.sweep_id and dist.is_initialized():
            if self.world_rank == 0:  # where the wandb agent has changed params
                objects = [self.params]
            else:
                self.params = None
                objects = [None]
            dist.broadcast_object_list(objects, src=0)
            self.params = objects[0]


        # logging
        # set_seed(self.params, self.world_size)
        if self.world_rank == 0:
            logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(exp_dir, 'out.log'))
            logging_utils.log_versions()


        # local batch size
        if not self.params.finetune:
            assert self.params.batch_size % self.world_size == 0, "The global_batch_size can not be divided by world_size"
        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = int(self.params.batch_size // self.world_size)


        # get dataloader, dataset, sampler
        # (-> utils.__init__ -> utils.data_loader_era5_dali -> utils.dali_era5_es_helper)
        # input: (params, files_pattern, distributed, train)
        #   -> params: swin config
        #   -> files_pattern: data dir
        #   -> distributed: distributed sampler (shuffle once to avoid conflict)
        #   -> train: bool (True or False)
        # return: (inp, tar, izen, tzen)
        self.train_data_loader, self.train_dataset, self.train_sampler = (get_data_loader_distributed(self.params,
                                                                                                      self.params.train_data_path,
                                                                                                      dist.is_initialized(),
                                                                                                      train=True,
                                                                                                      ))
        self.valid_data_loader, self.valid_dataset = (get_data_loader_distributed(self.params,
                                                                                  self.params.valid_data_path,
                                                                                  dist.is_initialized(),
                                                                                  train=False,
                                                                                  ))

        # (256, 320)
        self.params['img_shape_x'] = self.train_dataset.img_shape_x
        self.params['img_shape_y'] = self.train_dataset.img_shape_y

        # dump the yaml used
        if self.world_rank == 0:
            hparams = ruamelDict()
            yaml = YAML()
            for key, value in self.params.params.items():
                hparams[str(key)] = value.tolist() if isinstance(value, np.ndarray) else value
                with open(os.path.join(self.params['experiment_dir'], 'hyperparams.yaml'), 'w') as hpfile:
                    yaml.dump(hparams, hpfile)


        # computing losses
        self.loss_obj = LossHandler(self.params).to(self.device)


        # get Swin-T-V2
        self.model = get_model(self.params).to(self.device)


        # preprocessing data
        self.preprocessor = PreProcessor(self.params, self.device).to(self.device)
        if self.log_to_wandb:
            wandb.watch(self.model)


        # get optimizer
        if self.params.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.params.lr,
                                              betas=(0.9, 0.95),
                                              fused=True,
                                              )
        elif self.params.optimizer_type == 'adamw':
            decay = []
            no_decay = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                lname = name.lower()
                if param.ndim == 1 or lname.endswith(".bias") or "norm" in lname or "logit_scale" in lname or "pos_embed" in lname:
                    no_decay.append(param)
                else:
                    decay.append(param)
            param_groups = [{"params": decay, "weight_decay": 0.05},
                            {"params": no_decay, "weight_decay": 0.0}]
            self.optimizer = torch.optim.AdamW(param_groups, #self.model.parameters(),
                                               lr=self.params.lr,
                                               betas=(0.9, 0.95),
                                               eps = 1e-8,
                                               fused=True,
                                               )
        elif self.params.optimizer_type == 'FusedLAMB':
            self.optimizer = optimizers.FusedLAMB(self.model.parameters(),
                                                  lr=self.params.lr,
                                                  max_grad_norm=5.,
                                                  )
        else:
            raise Exception(f"optimizer type {self.params.optimizer_type} not implemented")


        # automatic mixed precision
        if self.params.enable_amp == True:
            self.gscaler = amp.GradScaler()


        # Distributed Data Parallel
        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                 device_ids=[self.local_rank],
                                                 output_device=[self.local_rank],
                                                 static_graph=(params.checkpointing > 0),
                                                 )


        # set ckpt attributes
        self.iters = 0
        self.startEpoch = 0


        # finetune
        if self.params.finetune and not self.params.resuming:
            assert (params.pretrained_checkpoint_path is not None), "error, please specify a valid pretrained checkpoint path"
            if self.log_to_screen:
                logging.info("Loading checkpoint %s" % self.params.pretrained_checkpoint_path)
            self.restore_checkpoint(params.pretrained_checkpoint_path)


        # resuming
        if self.params.resuming:
            if self.log_to_screen:
                logging.info("Loading checkpoint %s" % self.params.checkpoint_path)
            self.restore_checkpoint(self.params.checkpoint_path)
        self.epoch = self.startEpoch


        # get learning rate scheduler
        if self.params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        factor=0.2,
                                                                        patience=5,
                                                                        mode='min',
                                                                        )
        elif self.params.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        T_max=self.params.max_epochs,
                                                                        eta_min=1e-6,
                                                                        last_epoch=self.startEpoch - 1,
                                                                        )
        elif self.params.scheduler == 'CosineAnnealingLR_warmup':
            if not self.params.finetune:
                warmup_epochs = 10
            else:
                warmup_epochs = 3
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.params.max_epochs - warmup_epochs,
                eta_min=1e-6,
                last_epoch=-1
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        elif self.params.scheduler == 'CosineLR':
            if not self.params.finetune:
                self.scheduler = CosineLRScheduler(self.optimizer,  # warmup (0 ~ 9 epoch) from lr_min to self.lr
                                                   t_initial=30,    # first cosine  (10 ~ 39 epoch) from self.lr to lr_min
                                                   cycle_mul=1,     # second cosine (40 ~ 69 epoch) from self.lr to lr_min
                                                   cycle_limit=3,   # third cosine  (70 ~ 99 epoch) from self.lr to lr_min
                                                   lr_min=1e-6,     # rest 5 epoch keep low (max_epoch + 5)
                                                   warmup_lr_init=1e-6,
                                                   warmup_t=10,     # 10 epoch from warmup_lr to self.lr
                                                   warmup_prefix=True,
                                                   )
            else:
                self.scheduler = CosineLRScheduler(self.optimizer,  # warmup (0 ~ 2 epoch) from lr_min to self.lr
                                                   t_initial=10,    # first cosine  (3 ~ 12 epoch) from self.lr to lr_min
                                                   cycle_mul=1,     # second cosine (13 ~ 22 epoch) from self.lr to lr_min
                                                   cycle_limit=3,   # third cosine  (23 ~ 32 epoch) from self.lr to lr_min
                                                   lr_min=1e-6,     # rest 2 epoch keep low (max_epoch + 2)
                                                   warmup_lr_init=1e-6,
                                                   warmup_t=3,      # 3 epoch from warmup_lr to self.lr
                                                   warmup_prefix=True,
                                                   )
        else:
            self.scheduler = None
            raise Exception(f"learning rate scheduler {self.params.scheduler} not implemented")



        # logging model
        num_p = self.count_parameters()
        if self.log_to_screen:
            # logging.info(self.model)
            logging.info("Number of parameters = {}".format(num_p))
            # physics loss
            logging.info("Physics Loss: {}".format(self.physics))


        # launch training
        self.train()


    ##################
    # start training #
    ##################
    def train(self):
        # logging training loop
        if self.log_to_screen:
            logging.info("Starting Training Loop...")

        # start epoch
        best_valid_loss = 1.e6  # 1000000
        for epoch in range(self.startEpoch, self.params.max_epochs):
            # train sampler
            if dist.is_initialized() and (self.train_sampler is not None):
                self.train_sampler.set_epoch(epoch)

            # calculate one epoch time
            start = time.time()


            # !!!!!!!!!!!!!!!!!!!!!!!!! #
            # quick check learning rate #
            # !!!!!!!!!!!!!!!!!!!!!!!!! #
            # start train
            tr_time, train_logs = self.train_one_epoch()

            # start validate
            valid_time, valid_logs = self.validate_one_epoch()

            # learning rate scheduler
            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()
            elif self.params.scheduler == 'CosineAnnealingLR_warmup':
                self.scheduler.step()
            elif self.params.scheduler == 'CosineLR':
                self.scheduler.step(epoch+1)

            # logging learning rate
            if self.log_to_wandb:
                for pg in self.optimizer.param_groups:
                    lr = pg['lr']
                wandb.log({'lr': lr})
            if self.log_to_screen:
                logging.info("Dynamic Learning Rate: {}".format(self.optimizer.param_groups[0]['lr']))

            # save checkpoint at the end of every epoch
            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    self.save_checkpoint(self.params.checkpoint_path)
                    if valid_logs['valid_loss'] <= best_valid_loss:
                        logging.info('Val loss improved from {} to {}'.format(best_valid_loss, valid_logs['valid_loss']))
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        best_valid_loss = valid_logs['valid_loss']

            # logging time and loss
            if self.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                logging.info('Training time = {}, Valid time = {}'.format(tr_time, valid_time))
                logging.info('Train loss: {}. Valid loss: {}'.format(train_logs['loss'], valid_logs['valid_loss']))


    ###################
    # train one epoch #
    ###################
    def train_one_epoch(self):
        # calculate train time
        tr_start = time.time()

        # set train mode
        self.epoch += 1
        self.loss_obj.epoch = self.epoch
        self.model.train()
        tr_loss = []
        tr_pct = []

        # train data loader
        for i, data in enumerate(self.train_data_loader, 0):
            # !!!!!!!!!!!!!!!!!!!!!!!! #
            #   must to be shut down   #
            # !!!!!!!!!!!!!!!!!!!!!!!! #
            # if i>=1:
            #     break
            # !!!!!!!!!!!!!!!!!!!!!!!! #
            #   must to be shut down   #
            # !!!!!!!!!!!!!!!!!!!!!!!! #
            # inp -> (1, 75, 256, 320)
            # tar -> (1, 71, 256, 320)
            # coszen -> (1, 1, 256, 320)
            # gen -> (1, 71, 256, 320)
            inp, tar, coszen = self.preprocessor(data)
            self.model.zero_grad()
            with amp.autocast(self.params.enable_amp):
                gen = self.model(inp, coszen=coszen).to(self.device, dtype=torch.float)
                # torch.save(gen, "./gen")
                # torch.save(tar, "./tar")
                # torch.save(inp, "./inp")
                # loss = self.loss_obj(gen, tar, inp)
                if not self.physics:
                    loss = self.loss_obj(gen, tar, inp)
                else:
                    loss, physics_pct = self.loss_obj(gen, tar, inp)
                    # print('loss: ' + str(loss))
                    # print('percent of physics to data: ' + str(physics_pct))
            # check DDP loss
            if not torch.isfinite(loss):
                print(f"[Rank {dist.get_rank()}] Non-finite loss detected at epoch {self.epoch}, iter {i}")
                if dist.is_initialized():
                    dist.barrier()
                raise RuntimeError("NaN or Inf loss detected")
            if self.params.enable_amp:
                self.gscaler.scale(loss).backward()
                self.gscaler.step(self.optimizer)
            else:
                loss.backward()
                self.optimizer.step()
            if self.params.enable_amp:
                self.gscaler.update()

            # all reduce
            if dist.is_initialized():
                dist.all_reduce(loss)

            # mean loss keep same in local ranks
            tr_loss.append(loss.item() / dist.get_world_size())
            # print(loss.item() / dist.get_world_size())
            tr_pct.append(np.array(physics_pct.item()).mean())
            # print(np.array(physics_pct.item()).mean())

        # logging loss
        tr_loss = np.array(tr_loss, dtype='float64')        # avoid inf
        # logging once with rank 0
        if self.log_to_screen:
            logging.info('Percent (Physics / Data): {}'.format(np.round(np.mean(tr_pct), 2)))
        logs = {'loss': np.mean(tr_loss)}
        if self.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        # calculate train time
        tr_time = time.time() - tr_start
        return tr_time, logs


    ######################
    # validate one epoch #
    ######################
    def validate_one_epoch(self):
        # calculate validate time
        valid_start = time.time()

        # set validate mode
        self.model.eval()

        # load global stds for outputting normalization
        mult = torch.as_tensor(np.load(self.params.global_stds_path)[0, self.params.out_channels, 0, 0]).to(self.device)


        valid_buff = torch.zeros(3, dtype=torch.float32, device=self.device)
        valid_loss = valid_buff[0].view(-1)
        valid_steps = valid_buff[2].view(-1)

        #
        valid_weighted_rmse = torch.zeros(self.params.n_out_channels, dtype=torch.float32, device=self.device)
        valid_weighted_acc = torch.zeros(self.params.n_out_channels, dtype=torch.float32, device=self.device)

        # random sample for vis
        sample_idx = np.random.randint(len(self.valid_data_loader))

        # validate data loader
        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader, 0):
                # !!!!!!!!!!!!!!!!!!!!!!!! #
                #   must to be shut down   #
                # !!!!!!!!!!!!!!!!!!!!!!!! #
                # if i >= 1:
                #     break
                # !!!!!!!!!!!!!!!!!!!!!!!! #
                #   must to be shut down   #
                # !!!!!!!!!!!!!!!!!!!!!!!! #
                inp, tar, coszen = self.preprocessor(data)
                gen = self.model(inp, coszen=coszen).to(self.device, dtype=torch.float)
                if not self.physics:
                    loss = self.loss_obj(gen, tar, inp)
                else:
                    loss, physics_pct = self.loss_obj(gen, tar, inp)
                valid_loss += loss
                valid_steps += 1.
                # compute metrics on final step of rollout when n_future > 1
                # TODO fix this for dali dataloader
                tar = tar[:, -self.params.n_out_channels:]
                gen = gen[:, -self.params.n_out_channels:]
                valid_weighted_rmse += weighted_rmse_torch(gen, tar)
                if (i == sample_idx) and self.log_to_wandb:
                    fields = [gen[0, 0].detach().cpu().numpy(), tar[0, 0].detach().cpu().numpy()]

        #
        if dist.is_initialized():
            dist.all_reduce(valid_buff)
            dist.all_reduce(valid_weighted_rmse)

        # divide by number of steps
        valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
        valid_weighted_rmse = valid_weighted_rmse / valid_buff[2]
        valid_weighted_rmse *= mult

        # download buffers
        valid_buff_cpu = valid_buff.detach().cpu().numpy()
        valid_weighted_rmse_cpu = valid_weighted_rmse.detach().cpu().numpy()

        valid_weighted_rmse = mult * torch.mean(valid_weighted_rmse, axis=0)
        logs = {'valid_loss': valid_buff_cpu[0]}


        # track specific variables
        if hasattr(self.params, 'track_channels'):
            idxes = [self.params.channel_names.index(varname) for varname in self.params.track_channels]
            track_channels = self.params.track_channels
        else:
            track_channels = ['u10m', 'v10m']
            idxes = [0, 1]

        #
        for idx, var in zip(idxes, track_channels):
            logs.update({f'valid_rmse_{var}': valid_weighted_rmse_cpu[idx]})

        # no need and save disk
        if self.log_to_wandb:
            # fig = vis(fields)
            # logs['vis'] = wandb.Image(fig)
            # plt.close(fig)
            wandb.log(logs, step=self.epoch)

        # calculate validate time
        valid_time = time.time() - valid_start
        return valid_time, logs


    #############
    # save ckpt #
    #############
    def save_checkpoint(self, checkpoint_path, model=None):
        if not model:
            model = self.model
        torch.save({'iters': self.iters,
                    'epoch': self.epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    },
                   checkpoint_path,
                   )


    ################
    # restore ckpt #
    ################
    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank))
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)
        if self.params.resuming:
            self.iters = checkpoint['iters']
            self.startEpoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)                            # export name
    parser.add_argument("--yaml_config", default='./config/swin.yaml', type=str)        # config path
    parser.add_argument("--config", default='swin_71var', type=str)                     # config name
    # automatic mixed precision (-> True, !-> False)
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--sweep_id", default=None, type=str, help='sweep config from ./configs/sweeps.yaml')
    args = parser.parse_args()
    # YParams -> config_path, model_name
    params = YParams(os.path.abspath(args.yaml_config), args.config)                        # Dir:utils.YParams.YParams
    trainer = Trainer(params, args)
    if args.sweep_id and trainer.world_rank == 0:
        wandb.agent(args.sweep_id,
                    function=trainer.build_and_launch,
                    count=1,
                    entity=trainer.params.entity,
                    project=trainer.params.project,
                    )
    else:
        trainer.build_and_launch()
    #
    if dist.is_initialized():
        dist.barrier()
    logging.info('DONE')
