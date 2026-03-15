#!/usr/bin/env bash
#SBATCH --output=out/slurm/%x-%j.out
#SBATCH --job-name="qwen3"
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB

# BASE=EleutherAI/pythia-70m
# BASE=EleutherAI/pythia-160m
# BASE=EleutherAI/pythia-410m
# BASE=EleutherAI/pythia-1b
# BASE=EleutherAI/pythia-1.4b
# BASE=EleutherAI/pythia-2.8b

BASE=Qwen/Qwen3.5-0.8B
# BASE=Qwen/Qwen3.5-2B
# BASE=Qwen/Qwen3.5-4B

# BASE=google/flan-t5-small
# BASE=google/flan-t5-base
# BASE=google/flan-t5-large
# BASE=google/flan-t5-xl
# BASE=google/flan-t5-xxl

NAME=$(echo $BASE | awk -F / '{print $2}')
DATASET=estner
TASK=seq-cls

BATCH_SIZE=32
WORKERS=4
EPOCHS=3

MLFLOW_TRACKING_URI=sqlite:///mlflow.db
LOG_LEVEL=DEBUG

uv sync

uv run cli --log-level $LOG_LEVEL fine-tune \
    --model $BASE \
    --run-name $NAME-ft-system-$DATASET \
    --task $TASK \
    --dataset $DATASET \
    --prompt-mode system \
    --no-head-only \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate 5e-5 \
    --no-grad-chkpts \
    --mlflow-tracking-uri $MLFLOW_TRACKING_URI

uv run cli --log-level $LOG_LEVEL prompt-tune \
    --model $BASE \
    --run-name $NAME-pt-pretrained-$DATASET \
    --task $TASK \
    --dataset $DATASET \
    --prefix-init pretrained \
    --workers $WORKERS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate 5e-4 \
    --no-grad-chkpts \
    --mlflow-tracking-uri $MLFLOW_TRACKING_URI
