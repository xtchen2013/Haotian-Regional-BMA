#!/bin/bash

export RANK=$SLURM_PROCID                                     # rank: process id in a task
export WORLD_RANK=$SLURM_PROCID                               # world_rank: process id
export LOCAL_RANK=$SLURM_LOCALID                              # local_rank: process relative id in a node
export WORLD_SIZE=$SLURM_NTASKS                               # world_size: number of processes
export MASTER_PORT=29500                                      # default from torch launcher
export WANDB_START_METHOD="thread"                            # WandB start method
export MASTER_ADDR=$(hostname)                                # 'localhost'
export NCCL_P2P_LEVEL=NVL                                     # use P2P by NVLink
image=nersc/pytorch:ngc-23.07-v0                              # pytorch ngc 23.07 v0
env=~/.local/perlmutter/nersc_pytorch_ngc-23.07-v0            # pytorch ngc 23.07 v0
config_file=./config/swin.yaml                                # config file
#config="swin_region_71var_4stages_32batch_large"                         # large
#run_num="ep_70_adam_cos_lr_1e_3"
#config="swin_region_71var_4stages_32batch_large_4step"                   # large 4 steps
#run_num="ep_15_adam_cos_lr_1e_4"
#config="swin_region_71var_4stages_32batch_large_physics"                 # large physics
#run_num="ep_70_adam_cos_warmup_lr_1e_3"
#config="swin_region_71var_4stages_32batch_large_physics_4step"           # large physics 4 steps
#run_num="ep_15_adam_cos_warmup_lr_1e_4"
#config="swin_region_71var_4stages_32batch_large_physics_adamw"           # large physics adamW
#run_num="ep_105_adamw_cos_warmup_restart_lr_1e_3"
#config="swin_region_71var_4stages_32batch_large_physics_adamw_4step"     # large physics adamW 4 steps
#run_num="ep_35_adamw_cos_warmup_restart_lr_1e_4"
config="swin_region_71var_4stages_32batch_large_physics_adamw_8step"     # large physics adamW 8 steps
run_num="ep_35_adamw_cos_warmup_restart_lr_1e_4"
#config="swin_region_71var_4stages_32batch_large_physics_adamw_12step"     # large physics adamW 12 steps
#run_num="ep_35_adamw_cos_warmup_restart_lr_1e_4"


#export OMP_NUM_THREADS=8
# nproc_per_node should be matched with batch size
# v100 16G x8
taskset -c 0-64 torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
# v100 32G x4
#taskset -c 0-64 torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
# srun:
# -n/--ntasks                    number of tasks
# -c/--cpus-per-task             number of CPUs in each task
# --gpus-per-node                number of GPUs in each node

# bash:
# -c                             read from string

# torchrun:
# --nproc_per_node               number of processes in each node
# --nnodes                       number of nodes
# --node_rank                    rank of node
# --master_addr                  master node IP
# –master_port                   master node port
