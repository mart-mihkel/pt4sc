import json
from pathlib import Path
from typing import cast

import torch
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
        system_prompt=params["system_prompt"],
        workers=workers,
    )

    logger.debug("load training args from checkpoint")
    args: TrainingArguments = torch.load(
        f"{checkpoint}/training_args.bin",
        weights_only=False,
    )

    args.report_to = "none"
    args.eval_strategy = "no"

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=init_collate_fn(tokenizer=tokenizer, task=params["task"]),
        compute_metrics=init_metrics_fn(tokenizer=tokenizer, task=params["task"]),
    )

    pred = trainer.predict(data["test"])  # type: ignore
    trainer.save_metrics("test", pred.metrics)
