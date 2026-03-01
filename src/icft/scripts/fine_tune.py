from typing import cast

from torch.nn import Module
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification

from icft.common import ICFTDataset, ICFTTask, PromptMode, freeze, init_data, train
from icft.datasets.multinerd import Multinerd
from icft.logging import logger


def _init_model(
    task: ICFTTask,
    head_only: bool,
    tokenizer: PreTrainedTokenizerFast,
    data: Multinerd,
    model_path: str,
) -> Module:
    if task == "seq2seq":
        logger.debug("load seq2seq pretrained model")
        model, info = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            output_loading_info=True,
        )
    elif task == "seq-cls":
        logger.debug("load seq-cls pretrained model")
        model, info = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            output_loading_info=True,
            num_labels=len(data.ID2TAG),
            id2label=data.ID2TAG,
            label2id=data.TAG2ID,
        )
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
    system_prompt: PromptMode,
    head_only: bool,
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

    data, compute_metrics, collator = init_data(
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

    logger.info("")
    logger.info("Thing         | %-36s |", "Value")
    logger.info("--------------+-" + 36 * "-" + "-+")
    logger.info("model         | %-36s |", model_path)
    logger.info("params        | %-36d |", total)
    logger.info("trainable     | %-36d |", trainable)
    logger.info("task          | %-36s |", task)
    logger.info("prompt        | %-36s |", system_prompt)
    logger.info("dataset       | %-36s |", dataset)
    logger.info("train samples | %-36d |", len(data.train))
    logger.info("eval samples  | %-36d |", len(data.eval))
    logger.info("test samples  | %-36d |", len(data.test))
    logger.info("batch size    | %-36d |", batch_size)
    logger.info("epochs        | %-36d |", epochs)
    logger.info("")

    train(
        model=model,
        data=data,
        collator=collator,
        compute_metrics=compute_metrics,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
    )
