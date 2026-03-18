from icft.common import (
    init_collate_fn,
    init_data,
    init_metrics_fn,
    init_pt_model,
    init_tokenizer,
    train,
)
from icft.logging import logger
from icft.types import DatasetName, PrefixInit, Task


def prompt_tune(
    model_path: str,
    run_name: str,
    task: Task,
    dataset: DatasetName,
    prefix_init: PrefixInit,
    workers: int,
    epochs: int,
    batch_size: int,
    effective_batch_size: int,
    learning_rate: float,
    grad_chkpts: bool,
    mlflow_tracking_uri: str | None,
):
    tokenizer = init_tokenizer(model_path=model_path)
    data, info = init_data(
        tokenizer=tokenizer,
        task=task,
        dataset=dataset,
        prompt_mode="none",
        workers=workers,
    )

    if dataset == "superglue":
        logger.warning("drop superglue test data, labels are private")
        data.pop("test")

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

    logger.debug("base model '%s'", model_path)
    logger.debug("dataset '%s'", dataset)
    logger.debug("task '%s'", task)
    logger.debug("prefix init '%s'", prefix_init)
    logger.debug("virtual tokens %d", model.prefix.shape[0])
    logger.debug("total parameters %d", total)
    logger.debug("trainable parameters %d", trainable)
    logger.debug("head parameters %d", trainable - prefix)
    logger.debug("prefix parameters %d", prefix)

    train(
        model=model,
        data=data,
        collate_fn=init_collate_fn(tokenizer=tokenizer, task=task),
        metrics_fn=init_metrics_fn(tokenizer=tokenizer, task=task),
        run_name=run_name,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        effective_batch_size=effective_batch_size,
        grad_chkpts=grad_chkpts,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
