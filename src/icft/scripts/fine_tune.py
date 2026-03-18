from icft.logging import logger
from icft.scripts.common import (
    DatasetName,
    PromptMode,
    Task,
    init_collate_fn,
    init_data,
    init_metrics_fn,
    init_model,
    init_tokenizer,
    train,
)


def fine_tune(
    model_path: str,
    run_name: str,
    task: Task,
    dataset: DatasetName,
    prompt_mode: PromptMode,
    head_only: bool,
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
        prompt_mode=prompt_mode,
        workers=workers,
    )

    if dataset == "superglue":
        logger.warning("drop superglue test data, labels are private")
        data.pop("test")

    model, _ = init_model(
        task=task,
        head_only=head_only,
        tokenizer=tokenizer,
        model_path=model_path,
        data_info=info,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug("model '%s'", model_path)
    logger.debug("dataset '%s'", dataset)
    logger.debug("task '%s'", task)
    logger.debug("system prompt mode '%s'", prompt_mode)
    logger.debug("total parameters %d", total)
    logger.debug("trainable parameters %d", trainable)

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
