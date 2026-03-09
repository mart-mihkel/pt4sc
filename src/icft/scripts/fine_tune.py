from typing import cast

from torch.nn import Module
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification

from icft.common import (
    ICFTDataset,
    ICFTPrompt,
    ICFTTask,
    freeze,
    init_collate_fn,
    init_data,
    init_metrics_fn,
    train,
)
from icft.datasets import Dataset
from icft.logging import logger


def _init_model(
    task: ICFTTask,
    head_only: bool,
    tokenizer: PreTrainedTokenizerFast,
    data: Dataset,
    model_path: str,
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
            num_labels=len(data.ID2LABEL),
            id2label=data.ID2LABEL,
            label2id=data.LABEL2ID,
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


def main(
    task: ICFTTask,
    dataset: ICFTDataset,
    system_prompt: ICFTPrompt,
    head_only: bool,
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
        system_prompt=system_prompt,
        workers=workers,
    )

    model = _init_model(
        task=task,
        head_only=head_only,
        tokenizer=tokenizer,
        data=data,
        model_path=model_path,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Key                     | %-24s |", "Value")
    logger.info("------------------------+-" + 24 * "-" + "-+")
    logger.info("model                   | %-24s |", model_path.split("/")[-1])
    logger.info("params                  | %-24d |", total)
    logger.info("trainable               | %-24d |", trainable)
    logger.info("task                    | %-24s |", task)
    logger.info("prompt                  | %-24s |", system_prompt)
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
