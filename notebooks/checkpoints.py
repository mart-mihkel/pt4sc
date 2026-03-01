import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    from typing import cast

    import torch
    from torch.nn import Parameter
    from transformers import (
        AutoTokenizer,
        PreTrainedTokenizerFast,
    )

    from icft.datasets.multinerd import Multinerd
    from icft.models.pt import PTModelConfig, PTModel
    from icft.scripts.prompt_tune import _init_pt_model as init_pt_model

    return (
        AutoTokenizer,
        Multinerd,
        PTModel,
        PTModelConfig,
        Parameter,
        PreTrainedTokenizerFast,
        cast,
        init_pt_model,
        torch,
    )


@app.cell
def _():
    task = "seq-cls"
    prefix_init = "pretrained"
    model_path = "jhu-clsp/mmBERT-base"
    workers = 8
    return model_path, prefix_init, task, workers


@app.cell
def _(
    AutoTokenizer,
    Multinerd,
    PreTrainedTokenizerFast,
    cast,
    init_pt_model,
    model_path,
    prefix_init,
    task,
    workers,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = cast(PreTrainedTokenizerFast, tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data = Multinerd(tokenizer=tokenizer, task=task, workers=workers)

    model = init_pt_model(
        task=task,
        model_path=model_path,
        tokenizer=tokenizer,
        prefix_init=prefix_init,
        data=data,
    )

    if model.base.config.pad_token_id is None:
        model.base.config.pad_token_id = tokenizer.pad_token_id
    return (model,)


@app.cell
def _(PTModelConfig, Parameter, model, torch):
    model.prefix = Parameter(torch.ones_like(model.prefix))
    model.save_pretrained("/tmp/test")
    config = PTModelConfig.from_pretrained("/tmp/test")
    config
    return (config,)


@app.cell
def _(PTModel, config):
    loaded_model = PTModel.from_pretrained(
        "/tmp/test",
        config=config,
    )

    loaded_model
    return (loaded_model,)


@app.cell
def _(loaded_model):
    loaded_model.prefix
    return


if __name__ == "__main__":
    app.run()
