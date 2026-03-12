import json
from pathlib import Path
from typing import cast

import torch
from mlflow import end_run, log_metrics, search_runs, start_run
from mlflow.entities import Run
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from icft.common import init_collate_fn, init_data, init_metrics_fn
from icft.logging import logger
from icft.models import PTModel, PTModelConfig


def main(checkpoint: str, workers: int = 4):
    logger.debug("register pt model")
    AutoConfig.register("pt", PTModelConfig)
    AutoModel.register(PTModelConfig, PTModel)

    logger.debug("load cli parameters from checkpoint")
    with open(Path(checkpoint).parent / "cli_params.json") as f:
        params = json.load(f)

    logger.debug("load tokenizer from checkpoint")
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained(checkpoint),
    )

    logger.debug("load model from checkpoint")
    model = AutoModel.from_pretrained(checkpoint)

    data, _ = init_data(
        tokenizer=tokenizer,
        task=params["task"],
        dataset=params["dataset"],
        prompt_mode=params["prompt_mode"],
        workers=workers,
    )

    logger.debug("load training args from checkpoint")
    args: TrainingArguments = torch.load(
        f"{checkpoint}/training_args.bin",
        weights_only=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        data_collator=init_collate_fn(tokenizer=tokenizer, task=params["task"]),
        compute_metrics=init_metrics_fn(tokenizer=tokenizer, task=params["task"]),
    )

    run_name = params["run_name"]
    runs = search_runs(
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        output_format="list",
    )

    runs = cast(list[Run], runs)
    if len(runs) == 0:
        raise ValueError(f"No run '{run_name}'")
    elif len(runs) > 1:
        logger.warning("found multiple runs '%s', picking one", run_name)
        run = runs[0]
    else:
        logger.info("found existing run '%s'", run_name)
        run = runs[0]

    start_run(run_id=run.info.run_id)
    metrics = trainer.evaluate(data["test"], metric_key_prefix="test")  # type: ignore
    log_metrics(metrics)
    end_run()
