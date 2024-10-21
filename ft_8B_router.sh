#!/bin/bash

#SBATCH --job-name ft_8B_router
#SBATCH -o bash_logs/8B/router.log
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem 200G
#SBATCH --time=2-00:00:00

python  finetune_router.py \
        -k \
        data_file=router/router_train.json \
        savestring=8B/router \
        trainer.train_len=50000 \
        trainer.test_len=25 \
        trainer.validation_dataset=router/router_val.json \
        optimizer.lr=1.e-4 \