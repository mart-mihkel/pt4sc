from typing import cast

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from icft.common import (
    ICFTDataset,
    ICFTPrompt,
    ICFTTask,
    init_collate_fn,
    init_data,
    init_metrics_fn,
    init_model,
    train,
)
from icft.logging import logger


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

    data, info = init_data(
        tokenizer=tokenizer,
        task=task,
        dataset=dataset,
        system_prompt=system_prompt,
        workers=workers,
    )

    model = init_model(
        task=task,
        head_only=head_only,
        tokenizer=tokenizer,
        model_path=model_path,
        data_info=info,
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
