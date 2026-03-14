#!/usr/bin/env bash
#SBATCH --output=out/slurm/%x-%j.out
#SBATCH --job-name="mmbert-small"
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --partition=gpu
#SBATCH --mem=64GB

set -euo pipefail
mkdir -p out

BASE=jhu-clsp/mmBERT-small
# BASE=jhu-clsp/mmBERT-base

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
# BASE=Qwen/Qwen2.5-1.5B

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

# export MLFLOW_TRACKING_URI=

MODEL=$(echo $BASE | awk -F / '{print $2}')
DATASET=multinerd
LOG_LEVEL=DEBUG
TASK=seq-cls
BATCH_SIZE=64
WORKERS=0
EPOCHS=1

uv sync

uv run cli --log-level $LOG_LEVEL fine-tune \
    --model $BASE \
    --run-name $MODEL-ft-system-$DATASET \
    --task $TASK \
    --dataset $DATASET \
    --prompt-mode system \
    --no-head-only \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr 5e-5 \
    --no-grad-chkpts \
    --mlflow-tracking

uv run cli --log-level $LOG_LEVEL prompt-tune \
    --model $BASE \
    --run-name $MODEL-pt-pretrained-$DATASET \
    --task $TASK \
    --dataset $DATASET \
    --prefix-init pretrained \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr 1e-3 \
    --no-grad-chkpts \
    --mlflow-tracking
