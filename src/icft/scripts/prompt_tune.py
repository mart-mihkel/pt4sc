from typing import cast

import torch
from torch.nn import Parameter
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
)

from icft.common import freeze, init_collate_fn, init_data, init_metrics_fn, train
from icft.datasets.multinerd import Multinerd
from icft.logging import logger
from icft.models.pt import PTModel, PTModelConfig
from icft.types import ICFTDataset, ICFTTask, PrefixInit


def _init_pt_model(
    task: ICFTTask,
    prefix_init: PrefixInit,
    tokenizer: PreTrainedTokenizerFast,
    data: Multinerd,
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

    system_ids = torch.tensor(data.system_tokens["input_ids"])
    if task == "seq2seq":
        logger.debug("load base model for seq2seq")
        base = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        info = {"missing_keys": set()}
        config = PTModelConfig(
            task=task,
            pretrained_model=model_path,
            num_virtual_tokens=len(system_ids),
        )
    elif task == "seq-cls":
        logger.debug("load base model for seq-cls")
        base, info = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            output_loading_info=True,
            num_labels=len(data.ID2TAG),
            id2label=data.ID2TAG,
            label2id=data.TAG2ID,
        )

        config = PTModelConfig(
            task=task,
            pretrained_model=model_path,
            num_virtual_tokens=len(system_ids),
            num_labels=len(data.ID2TAG),
            id2label=cast(dict[int, str], data.ID2TAG),
            label2id=cast(dict[str, int], data.TAG2ID),
        )
    elif task == "causal-lm":
        logger.debug("load base model for causal-lm")
        base = AutoModelForCausalLM.from_pretrained(model_path)
        info = {"missing_keys": set()}
        config = PTModelConfig(
            task=task,
            pretrained_model=model_path,
            num_virtual_tokens=len(system_ids),
        )
    else:
        raise NotImplementedError(f"Task '{task}'")

    logger.debug("init pt-model")
    model = PTModel(config=config)

    logger.debug("load pretrained weights")
    model.base.load_state_dict(base.state_dict(), strict=False)

    freeze(model=model.base, skip=info["missing_keys"])

    emb = model.base.get_input_embeddings()
    if prefix_init == "random":
        logger.debug("init random prefix with %d tokens", len(system_ids))
        model.prefix = Parameter(torch.randn(1, len(system_ids), emb.embedding_dim))
    elif prefix_init == "pretrained":
        logger.debug("init pretrained prefix with %d tokens", len(system_ids))
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
    workers: int,
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

    logger.info("")
    logger.info("Thing          | %-36s |", "Value")
    logger.info("---------------+-" + 36 * "-" + "-+")
    logger.info("model          | %-36s |", model_path)
    logger.info("virtual tokens | %-36d |", model.prefix.shape[0])
    logger.info("params         | %-36d |", total)
    logger.info("trainable      | %-36d |", trainable)
    logger.info("prefix         | %-36d |", prefix)
    logger.info("head           | %-36d |", trainable - prefix)
    logger.info("task           | %-36s |", task)
    logger.info("prompt         | %-36s |", "none")
    logger.info("dataset        | %-36s |", dataset)
    logger.info("train samples  | %-36d |", len(data.train))
    logger.info("eval samples   | %-36d |", len(data.eval))
    logger.info("test samples   | %-36d |", len(data.test))
    logger.info("batch size     | %-36d |", batch_size)
    logger.info("epochs         | %-36d |", epochs)
    logger.info("")

    train(
        model=model,
        data=data,
        collate_fn=init_collate_fn(tokenizer=tokenizer, task=task),
        metrics_fn=init_metrics_fn(tokenizer=tokenizer, task=task),
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
    )
