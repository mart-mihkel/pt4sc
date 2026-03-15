from typing import cast

from rich.table import Table
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
from icft.logging import console
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
    lr: float,
    grad_chkpts: bool,
    mlflow_tracking: bool,
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
        prompt_mode="none",
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

    table = Table("Task parameter", "Value")
    table.add_row("Model", model_path)
    table.add_row("Dataset", dataset)
    table.add_row("Task", task)
    table.add_section()
    table.add_row("Prefix initialization", prefix_init)
    table.add_row("Virtual tokens", str(model.prefix.shape[0]))
    table.add_section()
    table.add_row("Total parameters", str(total))
    table.add_row("Trainable parameters", str(trainable))
    table.add_row("Head parameters", str(trainable - prefix))
    table.add_row("Prefix parameters", str(prefix))
    console.print(table)

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
        mlflow_tracking=mlflow_tracking,
    )
