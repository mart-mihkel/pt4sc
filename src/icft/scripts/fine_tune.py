import os

from rich.table import Table

from icft.common import (
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
from icft.logging import console, logger


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
    learning_rate: float,
    grad_chkpts: bool,
    mlflow_tracking_uri: str | None,
):
    os.makedirs("out", exist_ok=True)
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

    model = init_model(
        task=task,
        head_only=head_only,
        tokenizer=tokenizer,
        model_path=model_path,
        data_info=info,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    table = Table("Task parameter", "Value")
    table.add_row("Model", model_path)
    table.add_row("Dataset", dataset)
    table.add_row("Task", task)
    table.add_section()
    table.add_row("System prompt mode", prompt_mode)
    table.add_row("Head only", str(head_only))
    table.add_section()
    table.add_row("Total parameters", str(total))
    table.add_row("Trainable parameters", str(trainable))
    console.print(table)

    train(
        model=model,
        data=data,
        collate_fn=init_collate_fn(tokenizer=tokenizer, task=task),
        metrics_fn=init_metrics_fn(tokenizer=tokenizer, task=task),
        run_name=run_name,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_chkpts=grad_chkpts,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
