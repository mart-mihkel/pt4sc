import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():

    import torch
    from torch.nn import Parameter
    from transformers import (
        AutoTokenizer,
    )

    from icft.models import PTModel, PTModelConfig
    from icft.scripts.prompt_tune import init_pt_model

    return (
        AutoTokenizer,
        PTModel,
        PTModelConfig,
        Parameter,
        init_pt_model,
        torch,
    )


@app.cell
def _():
    task = "seq-cls"
    prefix_init = "pretrained"
    model_path = "jhu-clsp/mmBERT-base"
    return model_path, prefix_init, task


@app.cell
def _(AutoTokenizer, init_pt_model, model_path, prefix_init, task):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = init_pt_model(
        task=task,
        model_path=model_path,
        tokenizer=tokenizer,
        prefix_init=prefix_init,
        sys_enc=tokenizer("System prompt"),
        id2label={0: "0"},
        label2id={"0": 0},
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
