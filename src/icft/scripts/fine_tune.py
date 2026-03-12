from typing import cast

from rich.table import Table
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from icft.common import (
    DatasetName,
    PromptMode,
    Task,
    init_collate_fn,
    init_data,
    init_metrics_fn,
    init_model,
    train,
)
from icft.logging import console


def ft(
    task: Task,
    dataset: DatasetName,
    prompt_mode: PromptMode,
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
        prompt_mode=prompt_mode,
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

    table = Table(caption="Task parameters", show_header=False)
    table.add_row("model", model_path.split("/")[-1])
    table.add_row("params", str(total))
    table.add_row("trainable", str(trainable))
    table.add_row("task", task)
    table.add_row("prompt", prompt_mode)
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
