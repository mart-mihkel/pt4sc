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


def pt(
    task: Task,
    dataset: DatasetName,
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

    table = Table(caption="Task parameters", show_header=False)
    table.add_row("model", model_path.split("/")[-1])
    table.add_row("params", str(total))
    table.add_row("trainable", str(trainable))
    table.add_row("head", str(trainable - prefix))
    table.add_row("prefix", str(prefix))
    table.add_row("prefix init", prefix_init)
    table.add_row("prefix tokens", str(model.prefix.shape[0]))
    table.add_row("task", task)
    table.add_row("prompt", "none")
    table.add_row("dataset", dataset)
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
    )
