#!/usr/bin/env bash
#SBATCH --output=out/slurm/test-%j.out
#SBATCH --job-name="test"
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB

BASE=distilbert/distilbert-base-cased
# BASE=jhu-clsp/mmBERT-small
# BASE=openai-community/gpt2
# BASE=EleutherAI/pythia-70m
# BASE=Qwen/Qwen2.5-0.5B
# BASE=google-t5/t5-small
# BASE=google/flan-t5-small

NAME=$(echo $BASE | awk -F / '{print $2}')
DATASET=superglue
TASK=seq-cls

BATCH_SIZE=32
WORKERS=4
EPOCHS=0

MLFLOW_TRACKING_URI=sqlite:///mlflow.db
LOG_LEVEL=DEBUG

uv sync

uv run cli --log-level $LOG_LEVEL fine-tune \
    --model $BASE \
    --run-name test \
    --task $TASK \
    --dataset $DATASET \
    --prompt-mode system \
    --no-head-only \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate 5e-5 \
    --no-grad-chkpts

uv run cli --log-level $LOG_LEVEL prompt-tune \
    --model $BASE \
    --run-name test \
    --task $TASK \
    --dataset $DATASET \
    --prefix-init pretrained \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate 1e-3 \
    --no-grad-chkpts
