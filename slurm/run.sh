#!/usr/bin/env bash
#SBATCH --job-name="gpt2-medium"
#SBATCH --output=out/slurm/%x-%j.out
#SBATCH --time=96:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB

set -euo pipefail
mkdir -p out

# BASE=distilbert/distilbert-base-uncased
# BASE=jhu-clsp/mmBERT-small
# BASE=jhu-clsp/mmBERT-base
# BASE=FacebookAI/roberta-large

# BASE=openai-community/gpt2
BASE=openai-community/gpt2-medium
# BASE=openai-community/gpt2-large
# BASE=openai-community/gpt2-xl

# BASE=google-t5/t5-small
# BASE=google-t5/t5-base
# BASE=google-t5/t5-large
# BASE=google-t5/t5-3b
# BASE=google-t5/t5-11b

MODEL=$(echo $BASE | awk -F / '{print $2}')
DATASET=multinerd
LOG_LEVEL=DEBUG
TASK=seq-cls
BATCH_SIZE=8
WORKERS=16
EPOCHS=1

make install

# fine-tune
LR=5e-5

uv run cli.py --log-level $LOG_LEVEL fine-tune \
    --task $TASK --dataset $DATASET --system-prompt none --no-head-only \
    --model $BASE --run-name $MODEL-ft-none-$DATASET --grad-chkpts \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH_SIZE --workers $WORKERS

uv run cli.py --log-level $LOG_LEVEL fine-tune \
    --task $TASK --dataset $DATASET --system-prompt ner --no-head-only \
    --model $BASE --run-name $MODEL-ft-ner-$DATASET --grad-chkpts \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH_SIZE --workers $WORKERS

uv run cli.py --log-level $LOG_LEVEL fine-tune \
    --task $TASK --dataset $DATASET --system-prompt random --no-head-only \
    --model $BASE --run-name $MODEL-ft-random-$DATASET --grad-chkpts \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH_SIZE --workers $WORKERS

# prompt-tune
LR=1e-3

uv run cli.py --log-level $LOG_LEVEL prompt-tune \
    --task $TASK --dataset $DATASET --prefix-init pretrained \
    --model $BASE --run-name $MODEL-pt-pretrained-$DATASET --grad-chkpts \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH_SIZE --workers $WORKERS

uv run cli.py --log-level $LOG_LEVEL prompt-tune \
    --task $TASK --dataset $DATASET --prefix-init random \
    --model $BASE --run-name $MODEL-pt-random-$DATASET --grad-chkpts \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH_SIZE --workers $WORKERS
