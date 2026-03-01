#!/usr/bin/env bash
#SBATCH --job-name="mmbert-base"
#SBATCH --output=out/slurm/%x-%j.out
#SBATCH --partition=main

set -euo pipefail
mkdir -p out

# BASE=distilbert/distilbert-base-uncased
# BASE=jhu-clsp/mmBERT-small
BASE=jhu-clsp/mmBERT-base

# BASE=openai-community/gpt2
# BASE=openai-community/gpt2-medium
# BASE=openai-community/gpt2-large
# BASE=openai-community/gpt2-xl

# BASE=google-t5/t5-small
# BASE=google-t5/t5-base
# BASE=google-t5/t5-large
# BASE=google-t5/t5-3b
# BASE=google-t5/t5-11b

MODEL_NAME=$(echo $BASE | awk -F / '{print $2}')
DATASET=multinerd
LOG_LEVEL=DEBUG
TASK=seq-cls

BATCH_SIZE=8
WORKERS=16
EPOCHS=1

nix-shell --argstr run "make install"

# fine-tune
nix-shell --argstr run "uv run cli.py --log-level $LOG_LEVEL fine-tune \
    --task $TASK --dataset $DATASET --system-prompt none --no-head-only \
    --model $BASE --run-name ft-none-$DATASET-$MODEL_NAME \
    --epochs $EPOCHS --batch-size $BATCH_SIZE --workers $WORKERS"

nix-shell --argstr run "uv run cli.py --log-level $LOG_LEVEL fine-tune \
    --task $TASK --dataset $DATASET --system-prompt ner --no-head-only \
    --model $BASE --run-name ft-ner-$DATASET-$MODEL_NAME \
    --epochs $EPOCHS --batch-size $BATCH_SIZE --workers $WORKERS"

nix-shell --argstr run "uv run cli.py --log-level $LOG_LEVEL fine-tune \
    --task $TASK --dataset $DATASET --system-prompt random --no-head-only \
    --model $BASE --run-name ft-random-$DATASET-$MODEL_NAME \
    --epochs $EPOCHS --batch-size $BATCH_SIZE --workers $WORKERS"

# prompt-tune
nix-shell --argstr run "uv run cli.py --log-level $LOG_LEVEL prompt-tune \
    --task $TASK --dataset $DATASET --prefix-init pretrained \
    --model $BASE --run-name pt-pretrained-$DATASET-$MODEL_NAME \
    --epochs $EPOCHS --batch-size $BATCH_SIZE --workers $WORKERS" 

nix-shell --argstr run "uv run cli.py --log-level $LOG_LEVEL prompt-tune \
    --task $TASK --dataset $DATASET --prefix-init random \
    --model $BASE --run-name pt-random-$DATASET-$MODEL_NAME \
    --epochs $EPOCHS --batch-size $BATCH_SIZE --workers $WORKERS"
