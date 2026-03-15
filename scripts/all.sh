#!/usr/bin/env bash
#SBATCH --output=out/slurm/%x-%j.out
#SBATCH --job-name="mmbert"
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB

BASE=jhu-clsp/mmBERT-small
# BASE=jhu-clsp/mmBERT-base

NAME=$(echo $BASE | awk -F / '{print $2}')
DATASET=estner
TASK=seq-cls

BATCH_SIZE=32
WORKERS=4
EPOCHS=3

MLFLOW_TRACKING_URI=sqlite:///mlflow.db
LOG_LEVEL=DEBUG

uv sync

for PROMPT_MODE in "system" "random" "none"; do
    uv run cli --log-level $LOG_LEVEL fine-tune \
        --model $BASE \
        --run-name $NAME-ft-$PROMPT_MODE-$DATASET \
        --task $TASK \
        --dataset $DATASET \
        --prompt-mode $PROMPT_MODE \
        --no-head-only \
        --workers $WORKERS \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --learning-rate 5e-5 \
        --no-grad-chkpts \
        --mlflow-tracking-uri $MLFLOW_TRACKING_URI
done

for PREFIX_INIT in "pretrained" "random"; do
    uv run cli --log-level $LOG_LEVEL prompt-tune \
        --model $BASE \
        --run-name $NAME-pt-$PREFIX_INIT-$DATASET \
        --task $TASK \
        --dataset $DATASET \
        --prefix-init $PREFIX_INIT \
        --workers $WORKERS \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --learning-rate 5e-4 \
        --no-grad-chkpts \
        --mlflow-tracking-uri $MLFLOW_TRACKING_URI
done
