from collections.abc import Callable
from functools import partial
from math import ceil
from typing import Any, cast

import torch
from datasets.dataset_dict import DatasetDict
from datasets.splits import Split
from mlflow import end_run, set_experiment, set_tracking_uri, start_run
from torch.nn import Module, Parameter
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollator,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from icft.datasets.estner import init_estner
from icft.datasets.multinerd import DatasetInfo, init_multinerd
from icft.datasets.superglue import init_superglue
from icft.logging import logger
from icft.metrics import (
    compute_metrics_causal_lm,
    compute_metrics_seq2seq,
    compute_metrics_seq_cls,
)
from icft.models import (
    PTDecoderModel,
    PTDecoderModelConfig,
    PTEncoderDecoderModel,
    PTEncoderDecoderModelConfig,
    PTEncoderModel,
    PTEncoderModelConfig,
    PTModel,
)
from icft.types import DatasetName, PrefixInit, PromptMode, Task


def init_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    logger.debug("init tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)

    if tokenizer.pad_token is None:
        logger.warning("tokenizer doesn't have a padding token, using eos")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def init_model(
    task: Task,
    head_only: bool,
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    data_info: DatasetInfo,
) -> tuple[PreTrainedModel, dict[str, set[str]]]:
    loading_info = {"missing_keys": set()}
    if task == "seq2seq":
        logger.debug("load seq2seq pretrained model %s", model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    elif task == "seq-cls":
        logger.debug("load seq-cls pretrained model %s", model_path)
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            output_loading_info=True,
            num_labels=len(data_info["id2label"]),
            id2label=data_info["id2label"],
            label2id=data_info["label2id"],
        )
    elif task == "causal-lm":
        logger.debug("load causal-lm pretrained model %s", model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        raise NotImplementedError(f"Task '{task}'")

    if model.config.pad_token_id is None:
        logger.warning("model doesn't have a padding token, using eos")
        model.config.pad_token_id = tokenizer.eos_token_id

    if head_only:
        freeze(model=model, skip=loading_info["missing_keys"])

    return model, loading_info


def load_pt_model(checkpoint: str) -> PTModel:
    logger.debug("load pt-model from checkpoint")
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    return AutoModel.from_pretrained(
        checkpoint,
        config=config,
        local_files_only=True,
    )


def init_pt_model(
    task: Task,
    prefix_init: PrefixInit,
    tokenizer: PreTrainedTokenizerFast,
    model_path: str,
    data_info: DatasetInfo,
) -> PTModel:
    if "checkpoint" in model_path:
        return load_pt_model(checkpoint=model_path)

    cls_token = tokenizer.cls_token or "" if task == "seq-cls" else ""
    sys = tokenizer(
        f"{cls_token}{data_info['system_prompt']}",
        add_special_tokens=False,
        truncation=True,
    )

    system_ids = torch.tensor(sys["input_ids"])
    num_virtual_tokens = len(system_ids)

    base, loading_info = init_model(
        task=task,
        head_only=False,
        tokenizer=tokenizer,
        model_path=model_path,
        data_info=data_info,
    )

    model_type = base.config.model_type
    if model_type in {"bert", "distilbert", "roberta", "eurobert", "modernbert"}:
        logger.debug("init pt encoder model")
        config = PTEncoderModelConfig(
            task=task,
            pretrained_model=model_path,
            num_virtual_tokens=num_virtual_tokens,
            num_labels=len(data_info["id2label"]),
            id2label=data_info["id2label"],
            label2id=data_info["label2id"],
        )

        model = PTEncoderModel(config=config)
    elif model_type in {"gpt2", "gpt_neox", "gemma", "qwen2"}:
        logger.debug("init pt decoder model")
        config = PTDecoderModelConfig(
            task=task,
            pretrained_model=model_path,
            num_virtual_tokens=num_virtual_tokens,
            num_labels=len(data_info["id2label"]),
            id2label=data_info["id2label"],
            label2id=data_info["label2id"],
        )

        model = PTDecoderModel(config=config)
    elif model_type in {"t5", "t5gemma", "t5gemma2"}:
        logger.debug("init pt encoder-decoder model")
        config = PTEncoderDecoderModelConfig(
            task=task,
            pretrained_model=model_path,
            num_virtual_tokens=num_virtual_tokens,
            num_labels=len(data_info["id2label"]),
            id2label=data_info["id2label"],
            label2id=data_info["label2id"],
        )

        model = PTEncoderDecoderModel(config=config)
    else:
        raise NotImplementedError(f"PT model for base '{model_type}'")

    logger.debug("load pretrained weights")
    model.base.load_state_dict(base.state_dict(), strict=False)
    freeze(model=model.base, skip=loading_info["missing_keys"])

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


def init_collate_fn(tokenizer: PreTrainedTokenizerFast, task: Task) -> DataCollator:
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
    task: Task,
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
    task: Task,
    dataset: DatasetName,
    prompt_mode: PromptMode,
    workers: int,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    logger.debug("init %s dataset", dataset)
    if dataset == "multinerd":
        data, info = init_multinerd(
            tokenizer=tokenizer,
            prompt_mode=prompt_mode,
            task=task,
            workers=workers,
            filter_en=True,
            split=split,
        )
    elif dataset == "estner":
        data, info = init_estner(
            tokenizer=tokenizer,
            prompt_mode=prompt_mode,
            task=task,
            workers=workers,
            split=split,
        )
    elif dataset == "superglue":
        data, info = init_superglue(
            tokenizer=tokenizer,
            prompt_mode=prompt_mode,
            task=task,
            workers=workers,
            split=split,
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
    metrics_fn: Callable[[EvalPrediction], dict[str, int | float]],
    run_name: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    effective_batch_size: int,
    grad_chkpts: bool,
    mlflow_tracking_uri: str | None,
):
    have_cuda = torch.cuda.is_available()
    optim = "adamw_8bit" if have_cuda else "adamw_torch_fused"
    grad_acc_steps = max(1, ceil(effective_batch_size / batch_size))
    actual_effective_batch_size = batch_size * grad_acc_steps
    train_steps = ceil(len(data["train"]) / actual_effective_batch_size) * epochs
    eval_steps = max(1, train_steps // 5)
    logging_steps = max(1, train_steps // 100)
    out_dir = f"out/{run_name}"

    logger.debug("%shave cuda", "" if have_cuda else "don't ")
    logger.debug(
        "batch size %d, effective batch size %d with %d gradient accumulation steps",
        batch_size,
        actual_effective_batch_size,
        grad_acc_steps,
    )

    logger.debug("%d train samples", len(data["train"]))
    logger.debug("%d dev samples", len(data["dev"]))
    if "test" in data:
        logger.debug("%d test samples", len(data["test"]))
    else:
        logger.debug("0 test samples")

    args = TrainingArguments(
        run_name=run_name,
        report_to="mlflow" if mlflow_tracking_uri else "none",
        output_dir=out_dir,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
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
        args=args,
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
    )

    if mlflow_tracking_uri is not None:
        logger.info(
            "tracking experiment 'icft' run '%s' at %s",
            run_name,
            mlflow_tracking_uri,
        )

        set_tracking_uri(mlflow_tracking_uri)
        set_experiment("icft")
        start_run(run_name=run_name)
    else:
        logger.warning("no experiment tracking configured")

    logger.debug("start trainer")
    trainer.train()

    if "test" in data:
        logger.debug("run test evalatuaion")
        test = cast(Dataset, data["test"])
        trainer.evaluate(test, metric_key_prefix="test")
    else:
        logger.warning("skip test evalatuaion, no data")

    logger.debug("save checkpoint to %s", out_dir)
    trainer.save_model(out_dir)

    if mlflow_tracking_uri is not None:
        end_run()
