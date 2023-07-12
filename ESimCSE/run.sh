#!/bin/bash
#SBATCH --output=/cluster/work/medinfmk/MedVLM/output/output_%J.log
#SBATCH --error=/cluster/work/medinfmk/MedVLM/error/error_%j.log
#SBATCH --job-name=fkknb                # create a short name for your job
#SBATCH --partition=gpu
#SBATCH --nodes=1                       # node count
#SBATCH --gres=gpu:rtx3090:1            # titan_rtx & geforce_rtx_3090 & tesla_v100 & geforce_rtx_2080_ti & rtx_a6000
#SBATCH --cpus-per-task=1               # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=128:00:00                # total run time limit (HH:MM:SS)

# Send more noteworthy information to the output log
echo "Started at:     $(date)"

source ~/.bashrc
source ~/.bashrc.xzheng
conda activate simcse

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export CUDA_VISIBLE_DEVICES=0
# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=1

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=8

model="bert-base-uncased"
momentum=0.995
bs=64
dup_type=bpe

case $model in
    "bert-base-uncased")
        lr=3e-5
        neg_size=160
        dropout=0.1
        dup_rate=0.32
        ;;
    "bert-large-uncased")
        lr=1e-5
        neg_size=160
        dropout=0.1
        dup_rate=0.32
        ;;
    "roberta-base")
        lr=1e-5
        neg_size=160
        dropout=0.1
        dup_rate=0.3
        ;;
    "roberta-large")
        lr=1e-5
        neg_size=128
        dropout=0.15
        dup_rate=0.28
        ;;
esac

python train.py \
        --model_name_or_path ${model} \
        --train_file data/wiki1m_for_simcse.txt \
        --output_dir result/esimcse-CL3-${model} \
        --num_train_epochs 1 \
        --per_device_train_batch_size ${bs} \
        --learning_rate ${lr} \
        --max_seq_length 32 \
        --evaluation_strategy steps \
        --metric_for_best_model stsb_spearman \
        --load_best_model_at_end \
        --eval_steps 125 \
        --pooler_type cls \
        --mlp_only_train \
        --overwrite_output_dir \
        --temp 0.05 \
        --use_kmeans \
        --cluster_loss_lambda 1 \
        --clusters 10 \
        --num_iters 10 \
        --do_train \
        --do_eval \
        --fp16 \
        --dropout ${dropout} \
        --neg_size ${neg_size} \
        --dup_type ${dup_type} \
        --dup_rate ${dup_rate} \
        --momentum ${momentum}

python evaluation.py \
        --model_name_or_path result/esimcse-${model} \
        --pooler cls_before_pooler \
        --task_set sts \
        --mode test

