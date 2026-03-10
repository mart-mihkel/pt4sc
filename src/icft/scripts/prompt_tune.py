from typing import cast

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from icft.common import (
    init_collate_fn,
    init_data,
    init_metrics_fn,
    init_pt_model,
    train,
)
from icft.logging import logger
from icft.types import ICFTDataset, ICFTTask, PrefixInit


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

    data, info = init_data(
        tokenizer=tokenizer,
        task=task,
        dataset=dataset,
        system_prompt="none",
        workers=workers,
    )

    model = init_pt_model(
        task=task,
        prefix_init=prefix_init,
        tokenizer=tokenizer,
        model_path=model_path,
        data_info=info,
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
