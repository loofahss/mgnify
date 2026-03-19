#!/bin/bash
#SBATCH --job-name=filter_mgy      # 任务名称
#SBATCH --output=%j.out        # 输出文件名
#SBATCH --error=%j.err 
#SBATCH --partition=cpu       # 分区名称
#SBATCH --nodes=1

python filter_large_proteins.py
