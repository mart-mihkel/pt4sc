#!/usr/bin/env bash
#SBATCH --job-name="gwen2"
#SBATCH --output=out/slurm/%x-%j.out
#SBATCH --time=96:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB

set -euo pipefail
mkdir -p out

# encoder

# BASE=jhu-clsp/mmBERT-small
# BASE=jhu-clsp/mmBERT-base

# decoder

# BASE=openai-community/gpt2
# BASE=openai-community/gpt2-medium
# BASE=openai-community/gpt2-large
# BASE=openai-community/gpt2-xl

# BASE=EleutherAI/pythia-70m
# BASE=EleutherAI/pythia-160m
# BASE=EleutherAI/pythia-410m
# BASE=EleutherAI/pythia-1b
# BASE=EleutherAI/pythia-1.4b

# BASE=Qwen/Qwen2.5-0.5B
BASE=Qwen/Qwen2.5-1.5B

# encoder-decoder

# BASE=google-t5/t5-small
# BASE=google-t5/t5-base
# BASE=google-t5/t5-large
# BASE=google-t5/t5-3b
# BASE=google-t5/t5-11b

# BASE=google/flan-t5-small
# BASE=google/flan-t5-base
# BASE=google/flan-t5-large
# BASE=google/flan-t5-xl
# BASE=google/flan-t5-xxl

MODEL=$(echo $BASE | awk -F / '{print $2}')
DATASET=multinerd
LOG_LEVEL=DEBUG
TASK=seq-cls
BATCH_SIZE=8
WORKERS=4
EPOCHS=1

uv sync

# fine-tune
LR=5e-5

uv run cli.py --log-level $LOG_LEVEL fine-tune \
    --task $TASK --dataset $DATASET --system-prompt system --no-head-only \
    --model $BASE --run-name $MODEL-ft-system-$DATASET --no-grad-chkpts \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH_SIZE --workers $WORKERS

# prompt-tune
LR=1e-3

uv run cli.py --log-level $LOG_LEVEL prompt-tune \
    --task $TASK --dataset $DATASET --prefix-init pretrained \
    --model $BASE --run-name $MODEL-pt-pretrained-$DATASET --no-grad-chkpts \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH_SIZE --workers $WORKERS
