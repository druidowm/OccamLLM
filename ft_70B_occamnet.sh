#!/bin/bash

#SBATCH --job-name ft_70B_occamnet
#SBATCH -o bash_logs/70B/occamnet.log
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem 200G
#SBATCH --time=2-00:00:00

python finetune_occamnet.py \
        -k \
        savestring=70B/occamnet_part_1 \
        llm.assistant_prompt="Answer = " \
        llm.version=Meta-Llama-3-70B-Instruct \

python finetune_occamnet.py \
        -k \
        savestring=70B/occamnet \
        trainer.train_len=28000 \
        occamnet.from_pt=checkpoints/70B/occamnet_part_1/STEP80000 \
        llm.version=Meta-Llama-3-70B-Instruct \
