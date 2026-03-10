from collections.abc import Callable
from functools import partial
from math import ceil
from typing import Any

import torch
from datasets.dataset_dict import DatasetDict
from torch.nn import Module, Parameter
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    DataCollator,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizerFast,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from icft.datasets.multinerd import DatasetInfo, init_multinerd
from icft.logging import logger
from icft.metrics import (
    compute_metrics_causal_lm,
    compute_metrics_seq2seq,
    compute_metrics_seq_cls,
)
from icft.models import (
    PTDecoderModel,
    PTEncoderDecoderModel,
    PTEncoderModel,
    PTModel,
    PTModelConfig,
)
from icft.types import ICFTDataset, ICFTPrompt, ICFTTask, PrefixInit


def init_model(
    task: ICFTTask,
    head_only: bool,
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    data_info: DatasetInfo,
) -> Module:
    if task == "seq2seq":
        logger.debug("load seq2seq pretrained model %s", model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        info = {"missing_keys": set()}
    elif task == "seq-cls":
        logger.debug("load seq-cls pretrained model %s", model_path)
        model, info = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            output_loading_info=True,
            num_labels=len(data_info["id2label"]),
            id2label=data_info["id2label"],
            label2id=data_info["label2id"],
        )
    elif task == "causal-lm":
        logger.debug("load causal-lm pretrained model %s", model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        info = {"missing_keys": set()}
    else:
        raise NotImplementedError(f"Task '{task}'")

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    if head_only:
        freeze(model=model, skip=info["missing_keys"])

    return model


def init_pt_model(
    task: ICFTTask,
    prefix_init: PrefixInit,
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    data_info: DatasetInfo,
) -> PTModel:
    if "checkpoint" in model_path:
        logger.debug("load pt-model from checkpoint")
        config = PTModelConfig.from_pretrained(model_path, local_files_only=True)
        return PTModel.from_pretrained(
            model_path,
            config=config,
            local_files_only=True,
        )

    cls_token = tokenizer.cls_token or "" if task == "seq-cls" else ""
    sys = tokenizer(
        f"{cls_token}{data_info['system_prompt']}",
        add_special_tokens=False,
    )
    system_ids = torch.tensor(sys["input_ids"])
    num_virtual_tokens = len(system_ids)

    if task == "seq2seq":
        logger.debug("load seq2seq base model %s", model_path)
        base = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        info = {"missing_keys": set()}
        config = PTModelConfig(
            task=task,
            pretrained_model=model_path,
            num_virtual_tokens=num_virtual_tokens,
        )
    elif task == "seq-cls":
        logger.debug("load seq-cls base model %s", model_path)
        base, info = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            output_loading_info=True,
            num_labels=len(data_info["id2label"]),
            id2label=data_info["id2label"],
            label2id=data_info["label2id"],
        )

        config = PTModelConfig(
            task=task,
            pretrained_model=model_path,
            num_virtual_tokens=num_virtual_tokens,
            num_labels=len(data_info["id2label"]),
            id2label=data_info["id2label"],
            label2id=data_info["label2id"],
        )
    elif task == "causal-lm":
        logger.debug("load causal-lm base model %s", model_path)
        base = AutoModelForCausalLM.from_pretrained(model_path)
        info = {"missing_keys": set()}
        config = PTModelConfig(
            task=task,
            pretrained_model=model_path,
            num_virtual_tokens=num_virtual_tokens,
        )
    else:
        raise NotImplementedError(f"Task '{task}'")

    model_type = base.config.model_type
    if model_type in {"gpt2", "gpt_neox", "gemma", "qwen2"}:
        logger.debug("init pt decoder model")
        model = PTDecoderModel(config=config)
    elif model_type in {"t5", "t5gemma", "t5gemma2"}:
        logger.debug("init pt encoder-decoder model")
        model = PTEncoderDecoderModel(config=config)
    elif model_type in {"bert", "distilbert", "roberta", "modernbert"}:
        logger.debug("init pt encoder model")
        model = PTEncoderModel(config=config)
    else:
        raise NotImplementedError("PT model for base '{model_type}'")

    logger.debug("load pretrained weights")
    model.base.load_state_dict(base.state_dict(), strict=False)

    freeze(model=model.base, skip=info["missing_keys"])

    emb = model.base.get_input_embeddings()
    if prefix_init == "random":
        logger.debug("init random prefix with %d tokens", num_virtual_tokens)
        model.prefix = Parameter(torch.randn(1, num_virtual_tokens, emb.embedding_dim))
    elif prefix_init == "pretrained":
        logger.debug("init pretrained prefix with %d tokens", num_virtual_tokens)
        model.prefix = Parameter(emb(system_ids).detach())
    else:
        raise NotImplementedError(f"Prefix init '{prefix_init}'")

    if model.base.config.pad_token_id is None:
        model.base.config.pad_token_id = tokenizer.pad_token_id

    return model


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
) -> tuple[DatasetDict, DatasetInfo]:
    logger.debug("init %s dataset", dataset)
    if dataset == "multinerd":
        data, info = init_multinerd(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            task=task,
            workers=workers,
            filter_en=True,
        )
    else:
        raise NotImplementedError(f"Dataset '{dataset}'")

    return data, info


def freeze(model: Module, skip: set[str]):
    logger.debug("freeze base model")
    for name, param in model.named_parameters():
        if name in skip:
            logger.debug("skip '%s'", name)
            continue

        param.requires_grad = False


def train(
    model: Module,
    data: DatasetDict,
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
    train_steps = ceil(len(data["train"]) / effective_batch_size) * epochs
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
    logger.info("train samples           | %-24d |", len(data["train"]))
    logger.info("eval samples            | %-24d |", len(data["validation"]))
    logger.info("test samples            | %-24d |", len(data["test"]))
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
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
    )

    trainer.train()
