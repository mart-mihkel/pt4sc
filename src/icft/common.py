from typing import Callable

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


def init_data(
    tokenizer: PreTrainedTokenizerFast,
    task: ICFTTask,
    dataset: ICFTDataset,
    system_prompt: PromptMode,
    workers: int,
) -> tuple[Multinerd, Callable, DataCollator]:
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
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        compute_metrics = data.compute_metrics_seq_cls  # TODO: seq2seq metrics
    elif task == "seq-cls":
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        compute_metrics = data.compute_metrics_seq_cls
    else:
        raise NotImplementedError(f"Task '{task}'")

    return data, compute_metrics, collator


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
    collator: DataCollator,
    compute_metrics: Callable,
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
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate(
        eval_dataset=data.test,
        metric_key_prefix="test",
    )

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)
