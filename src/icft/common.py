from functools import partial
from math import ceil
from typing import Any, Callable

import torch
from torch.nn import Module
from transformers import (
    DataCollator,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizerFast,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from icft.datasets import Dataset
from icft.datasets.multinerd import Multinerd
from icft.logging import logger
from icft.metrics import (
    compute_metrics_causal_lm,
    compute_metrics_seq2seq,
    compute_metrics_seq_cls,
)
from icft.types import ICFTDataset, ICFTPrompt, ICFTTask


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


def init_metrics_fn(
    task: ICFTTask,
    tokenizer: PreTrainedTokenizerFast | None = None,
) -> Callable[[EvalPrediction], dict[str, int | float]]:
    if task == "seq2seq":
        if tokenizer is None:
            raise ValueError("Tokenizer required for seq2seq metrics")

        return partial(compute_metrics_seq2seq, tokenizer=tokenizer)
    elif task == "seq-cls":
        return compute_metrics_seq_cls
    elif task == "causal-lm":
        if tokenizer is None:
            raise ValueError("Tokenizer required for causal-lm metrics")

        return partial(compute_metrics_causal_lm, tokenizer=tokenizer)
    else:
        raise NotImplementedError(f"Task '{task}'")


def init_data(
    tokenizer: PreTrainedTokenizerFast,
    task: ICFTTask,
    dataset: ICFTDataset,
    system_prompt: ICFTPrompt,
    workers: int,
) -> Dataset:
    logger.debug("init %s dataset", dataset)
    if dataset == "multinerd":
        data = Multinerd(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            task=task,
            workers=workers,
        )
    else:
        raise NotImplementedError(f"Dataset '{dataset}'")

    return data


def freeze(model: Module, skip: set[str]):
    logger.debug("freeze base model")
    for name, param in model.named_parameters():
        if name in skip:
            logger.debug("skip '%s'", name)
            continue

        param.requires_grad = False


def train(
    model: Module,
    data: Dataset,
    collate_fn: DataCollator,
    metrics_fn: Callable,
    run_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_chkpts: bool,
):
    have_cuda = torch.cuda.is_available()
    optim = "adamw_8bit" if have_cuda else "adamw_torch_fused"
    grad_acc_steps = max(1, ceil(64 / batch_size))
    effective_batch_size = batch_size * grad_acc_steps
    train_steps = ceil(len(data.train) / effective_batch_size) * epochs
    eval_steps = max(1, train_steps // 5)
    logging_steps = max(1, train_steps // 100)

    logger.info("CUDA                    | %-24s |", have_cuda)
    logger.info("optimizer               | %-24s |", optim)
    logger.info("epochs                  | %-24d |", epochs)
    logger.info("learning rate           | %-24f |", lr)
    logger.info("grad checkpoints        | %-24s |", grad_chkpts)
    logger.info("grad accumulation steps | %-24d |", grad_acc_steps)
    logger.info("batch size              | %-24d |", batch_size)
    logger.info("effective batch size    | %-24d |", effective_batch_size)
    logger.info("train samples           | %-24d |", len(data.train))
    logger.info("eval samples            | %-24d |", len(data.eval))
    logger.info("test samples            | %-24d |", len(data.test))
    logger.info("train steps             | %-24d |", train_steps)
    logger.info("logging steps           | %-24d |", logging_steps)
    logger.info("eval steps              | %-24d |", eval_steps)

    logger.debug("init trainer")

    args = TrainingArguments(
        run_name=run_name,
        report_to="mlflow",
        output_dir=f"out/{run_name}",
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        learning_rate=lr,
        optim=optim,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 4,
        gradient_accumulation_steps=grad_acc_steps,
        gradient_checkpointing=grad_chkpts,
        bf16_full_eval=have_cuda,
        bf16=have_cuda,
        fp16_full_eval=not have_cuda,
        fp16=not have_cuda,
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
