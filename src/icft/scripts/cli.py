from collections.abc import Callable
from typing import Annotated, Any, Literal

from typer import Context, Option, Typer

from icft.types import DatasetName, PrefixInit, PromptMode, Task

app = Typer(no_args_is_help=True, add_completion=False)


def timed(func: Callable) -> Callable:
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        from icft.logging import logger

        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info("time elapsed %02d:%02d:%02d", hours, minutes, seconds)

        return result

    return wrapper


def save_params(params: dict[str, Any], run_name: str):
    import json
    import os

    os.makedirs(f"out/{run_name}", exist_ok=True)
    with open(f"out/{run_name}/cli_params.json", "w") as f:
        json.dump(params, f, indent=2)


@app.callback()
def callback(log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"):
    from icft.logging import logger

    logger.setLevel(log_level)


@app.command()
@timed
def fine_tune(
    ctx: Context,
    model: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    task: Annotated[Task.__value__, Option()],
    dataset: Annotated[DatasetName.__value__, Option()],
    prompt_mode: Annotated[PromptMode.__value__, Option()],
    head_only: Annotated[bool, Option()],
    workers: int = 0,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 5e-5,
    grad_chkpts: bool = False,
    mlflow_tracking: bool = False,
):
    from icft.scripts.fine_tune import fine_tune

    save_params(ctx.params, run_name)
    fine_tune(
        model_path=model,
        run_name=run_name,
        task=task,
        dataset=dataset,
        prompt_mode=prompt_mode,
        head_only=head_only,
        workers=workers,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        grad_chkpts=grad_chkpts,
        mlflow_tracking=mlflow_tracking,
    )


@app.command()
@timed
def prompt_tune(
    ctx: Context,
    model: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    task: Annotated[Task.__value__, Option()],
    dataset: Annotated[DatasetName.__value__, Option()],
    prefix_init: Annotated[PrefixInit.__value__, Option()],
    workers: int = 0,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 5e-5,
    grad_chkpts: bool = False,
    mlflow_tracking: bool = False,
):
    from icft.scripts.prompt_tune import prompt_tune

    save_params(ctx.params, run_name)
    prompt_tune(
        model_path=model,
        run_name=run_name,
        task=task,
        dataset=dataset,
        prefix_init=prefix_init,
        workers=workers,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        grad_chkpts=grad_chkpts,
        mlflow_tracking=mlflow_tracking,
    )


if __name__ == "__main__":
    app()
