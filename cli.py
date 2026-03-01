from typing import Annotated, Literal

from typer import Option, Typer

app = Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def callback(log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"):
    from icft.logging import logger

    logger.setLevel(log_level)


@app.command()
def fine_tune(
    task: Annotated[Literal["seq2seq", "seq-cls", "causal-lm"], Option()],
    dataset: Annotated[Literal["multinerd"], Option()],
    system_prompt: Annotated[Literal["ner", "random", "none"], Option()],
    head_only: Annotated[bool, Option()],
    model: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    epochs: int = 1,
    batch_size: int = 8,
    workers: int = 4,
):
    from icft.scripts.fine_tune import main

    main(
        task=task,
        dataset=dataset,
        system_prompt=system_prompt,
        head_only=head_only,
        model_path=model,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
    )


@app.command()
def prompt_tune(
    task: Annotated[Literal["seq2seq", "seq-cls", "causal-lm"], Option()],
    dataset: Annotated[Literal["multinerd"], Option()],
    prefix_init: Annotated[Literal["pretrained", "labels", "random"], Option()],
    model: Annotated[str, Option()],
    run_name: Annotated[str, Option()],
    epochs: int = 1,
    batch_size: int = 8,
    workers: int = 4,
):
    from icft.scripts.prompt_tune import main

    main(
        task=task,
        dataset=dataset,
        prefix_init=prefix_init,
        model_path=model,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
    )


@app.command()
def collect_metrics():
    import json
    import os

    import polars as pl

    from icft.logging import logger

    records = []
    for run in os.listdir("out"):
        path = f"out/{run}/test_results.json"
        if not os.path.exists(path):
            continue

        with open(path) as f:
            res = json.load(f)

        res["run_name"] = run
        records.append(res)

    out = "out/test_results.json"
    df = pl.from_dicts(records)
    df.write_json(out)

    logger.info(df)
    logger.info("saved to '%s'", out)


if __name__ == "__main__":
    app()
