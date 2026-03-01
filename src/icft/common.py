from typing import Any, Callable

import torch
from torch.nn import Module
from transformers import (
    DataCollator,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from icft.datasets.multinerd import Multinerd
from icft.logging import logger
from icft.types import ICFTDataset, ICFTTask, PromptMode


def init_collate_fn(tokenizer: PreTrainedTokenizerFast, task: ICFTTask) -> DataCollator:
    _padding_collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    def _causal_lm_collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
        labels = [torch.tensor(f["labels"]) for f in features]
        no_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        out_features = _padding_collate_fn(no_labels)
        input_len = out_features["input_ids"].size(1)

        padded_labels = []
        for label in labels:
            pad_len = input_len - label.size(0)
            padded = torch.cat([label, torch.full((pad_len,), -100)])
            padded_labels.append(padded)

        out_features["labels"] = torch.stack(padded_labels)
        return out_features

    if task == "seq2seq":
        return DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)
    elif task == "seq-cls":
        return _padding_collate_fn
    elif task == "causal-lm":
        return _causal_lm_collate_fn
    else:
        raise NotImplementedError(f"Task '{task}'")


def init_data(
    tokenizer: PreTrainedTokenizerFast,
    task: ICFTTask,
    dataset: ICFTDataset,
    system_prompt: PromptMode,
    workers: int,
) -> tuple[Multinerd, Callable]:
    if dataset == "multinerd":
        logger.debug("init multinerd")
        data = Multinerd(
            tokenizer=tokenizer,
            system_prompt_mode=system_prompt,
            task=task,
            workers=workers,
        )
    else:
        raise NotImplementedError(f"Dataset '{dataset}'")

    if task == "seq2seq":
        metrics_fn = data.compute_metrics_seq_cls  # TODO: seq2seq metrics
    elif task == "seq-cls":
        metrics_fn = data.compute_metrics_seq_cls
    elif task == "causal-lm":
        metrics_fn = data.compute_metrics_seq_cls  # TODO: causal-lm metrics
    else:
        raise NotImplementedError(f"Task '{task}'")

    return data, metrics_fn


def freeze(model: Module, skip: set[str]):
    logger.debug("freeze base model")
    for name, param in model.named_parameters():
        if name in skip:
            logger.debug("skip '%s'", name)
            continue

        param.requires_grad = False


def train(
    model: Module,
    data: Multinerd,
    collate_fn: DataCollator,
    metrics_fn: Callable,
    run_name: str,
    epochs: int,
    batch_size: int,
):
    logger.debug("init trainer")
    steps_per_epoch = len(data.train) // (batch_size * epochs)
    args = TrainingArguments(
        project="icft",
        output_dir=f"out/{run_name}",
        report_to="trackio",
        trackio_space_id=None,
        run_name=run_name,
        logging_steps=steps_per_epoch // 100,
        eval_steps=steps_per_epoch // 5,
        eval_strategy="steps",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 4,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        fp16=True,
    )

    trainer = Trainer(
        model,
        args=args,
        train_dataset=data.train,
        eval_dataset=data.eval,
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
    )

    trainer.train()

    metrics = trainer.evaluate(
        eval_dataset=data.test,
        metric_key_prefix="test",
    )

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)
