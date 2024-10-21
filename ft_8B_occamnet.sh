#!/bin/bash

#SBATCH --job-name ft_8B_occamnet
#SBATCH -o bash_logs/8B/occamnet.log
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem 200G
#SBATCH --time=2-00:00:00

python finetune_occamnet.py \
        -k \
        savestring=8B/occamnet_part_1 \
        llm.assistant_prompt="Answer = " \

python finetune_occamnet.py \
        -k \
        savestring=8B/occamnet \
        trainer.train_len=28000 \
        occamnet.from_pt=checkpoints/8B/occamnet_part_1/STEP80000 \