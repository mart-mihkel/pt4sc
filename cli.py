from typing import Annotated, Any, Callable, Literal

from typer import Context, Option, Typer

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
    task: Annotated[Literal["seq2seq", "seq-cls", "causal-lm"], Option()],
    dataset: Annotated[Literal["multinerd"], Option()],
    system_prompt: Annotated[Literal["system", "random", "none"], Option()],
    head_only: Annotated[bool, Option()],
    model: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    grad_chkpts: bool = False,
    epochs: int = 1,
    lr: float = 5e-5,
    batch_size: int = 8,
    workers: int = 4,
):
    from icft.scripts.fine_tune import main

    save_params(ctx.params, run_name)
    main(
        task=task,
        dataset=dataset,
        system_prompt=system_prompt,
        head_only=head_only,
        model_path=model,
        run_name=run_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        workers=workers,
        grad_chkpts=grad_chkpts,
    )


@app.command()
@timed
def prompt_tune(
    ctx: Context,
    task: Annotated[Literal["seq2seq", "seq-cls", "causal-lm"], Option()],
    dataset: Annotated[Literal["multinerd"], Option()],
    prefix_init: Annotated[Literal["pretrained", "labels", "random"], Option()],
    model: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    grad_chkpts: bool = False,
    epochs: int = 1,
    lr: float = 5e-5,
    batch_size: int = 8,
    workers: int = 4,
):
    from icft.scripts.prompt_tune import main

    save_params(ctx.params, run_name)
    main(
        task=task,
        dataset=dataset,
        prefix_init=prefix_init,
        model_path=model,
        run_name=run_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        workers=workers,
        grad_chkpts=grad_chkpts,
    )


@app.command()
@timed
def predict(checkpoint: Annotated[str, Option()], workers: int = 4):
    from icft.scripts.predict import main

    main(checkpoint=checkpoint, workers=workers)


if __name__ == "__main__":
    app()
