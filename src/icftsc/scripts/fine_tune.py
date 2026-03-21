from transformers import AutoConfig

from icftsc.logging import logger
from icftsc.scripts.common import (
    init_collator,
    init_data,
    init_metrics_fn,
    init_model,
    init_tokenizer,
    train,
)
from icftsc.types import DatasetName, Task


def fine_tune(
    model_path: str,
    run_name: str,
    dataset: DatasetName,
    task: Task,
    head_only: bool,
    workers: int,
    epochs: int,
    batch_size: int,
    effective_batch_size: int,
    learning_rate: float,
    grad_chkpts: bool,
    mlflow_tracking_uri: str | None,
):
    logger.info("load model config")
    config = AutoConfig.from_pretrained(model_path)

    logger.info("load pretrained tokenizer")
    tokenizer = init_tokenizer(model_path=model_path)
    collate_fn = init_collator(tokenizer=tokenizer, task=task)
    metrics_fn = init_metrics_fn(task=task, tokenizer=tokenizer)

    logger.info("load dataset '%s'", dataset)
    data, info = init_data(
        model_type=config.model_type,
        tokenizer=tokenizer,
        dataset=dataset,
        workers=workers,
        task=task,
    )

    if dataset == "superglue-boolq":
        logger.warning("drop superglue test data, labels are private")
        data.pop("test")
        data.pop("test-system")
        data.pop("test-random")

    logger.info("load pretrained '%s' for '%s'", model_path, task)
    model, _ = init_model(
        head_only=head_only,
        tokenizer=tokenizer,
        model_path=model_path,
        data_info=info,
        task=task,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("total parameters %d", total)
    logger.info("trainable parameters %d", trainable)

    train(
        model=model,
        data=data,
        collate_fn=collate_fn,
        metrics_fn=metrics_fn,
        run_name=run_name,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        effective_batch_size=effective_batch_size,
        grad_chkpts=grad_chkpts,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
