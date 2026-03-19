#!/bin/bash
#SBATCH --job-name=predict_mgy      # 任务名称
#SBATCH --output=%j.out        # 输出文件名
#SBATCH --error=%j.err 
#SBATCH --partition=gpu       # 分区名称
#SBATCH --nodes=1

python predict_mgy.py \
--fasta /home/wangjy/code/mgnify/chunks/cleaned_proteins_filtered.part_001.fa.gz \
--model /home/wangjy/code/signalP/best_model_fold0.pth \
--output /home/wangjy/code/mgnify/pys/predictions/test_prediction.tsv \
--batch_size 64
