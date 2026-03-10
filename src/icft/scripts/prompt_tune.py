from typing import cast

import torch
from torch.nn import Parameter
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from icft.common import freeze, init_collate_fn, init_data, init_metrics_fn, train
from icft.datasets import Dataset
from icft.logging import logger
from icft.models import (
    PTDecoderModel,
    PTEncoderDecoderModel,
    PTEncoderModel,
    PTModel,
    PTModelConfig,
)
from icft.types import ICFTDataset, ICFTTask, PrefixInit


def _init_pt_model(
    task: ICFTTask,
    prefix_init: PrefixInit,
    tokenizer: PreTrainedTokenizerFast,
    data: Dataset,
    model_path: str,
) -> PTModel:
    if "checkpoint" in model_path:
        logger.debug("load pt-model from checkpoint")
        config = PTModelConfig.from_pretrained(model_path, local_files_only=True)
        return PTModel.from_pretrained(
            model_path,
            config=config,
            local_files_only=True,
        )

    system_ids = torch.tensor(data.system_prompt_tokens["input_ids"])
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
            num_labels=len(data.ID2LABEL),
            id2label=data.ID2LABEL,
            label2id=data.LABEL2ID,
        )

        config = PTModelConfig(
            task=task,
            pretrained_model=model_path,
            num_virtual_tokens=num_virtual_tokens,
            num_labels=len(data.ID2LABEL),
            id2label=data.ID2LABEL,
            label2id=data.LABEL2ID,
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


def main(
    task: ICFTTask,
    dataset: ICFTDataset,
    prefix_init: PrefixInit,
    model_path: str,
    run_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    workers: int,
    grad_chkpts: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data = init_data(
        tokenizer=tokenizer,
        task=task,
        dataset=dataset,
        system_prompt="none",
        workers=workers,
    )

    model = _init_pt_model(
        task=task,
        prefix_init=prefix_init,
        tokenizer=tokenizer,
        data=data,
        model_path=model_path,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    prefix = model.prefix.numel()

    logger.info("Key                     | %-24s |", "Value")
    logger.info("------------------------+-" + 24 * "-" + "-+")
    logger.info("model                   | %-24s |", model_path.split("/")[-1])
    logger.info("params                  | %-24d |", total)
    logger.info("trainable               | %-24d |", trainable)
    logger.info("head                    | %-24d |", trainable - prefix)
    logger.info("prefix                  | %-24d |", prefix)
    logger.info("prefix init             | %-24s |", prefix_init)
    logger.info("prefix tokens           | %-24d |", model.prefix.shape[0])
    logger.info("task                    | %-24s |", task)
    logger.info("prompt                  | %-24s |", "none")
    logger.info("dataset                 | %-24s |", dataset)

    train(
        model=model,
        data=data,
        collate_fn=init_collate_fn(tokenizer=tokenizer, task=task),
        metrics_fn=init_metrics_fn(tokenizer=tokenizer, task=task),
        run_name=run_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        grad_chkpts=grad_chkpts,
    )
