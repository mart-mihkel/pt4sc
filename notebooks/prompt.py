import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    from transformers import AutoTokenizer

    from icft.datasets.multinerd import init_multinerd

    return AutoTokenizer, init_multinerd


@app.cell
def _():
    # task = "seq-cls"
    # pretrained_model = "jhu-clsp/mmBERT-base"

    # task = "seq2seq"
    # pretrained_model = "google-t5/t5-small"

    task = "causal-lm"
    pretrained_model = "openai-community/gpt2"
    return pretrained_model, task


@app.cell
def _(AutoTokenizer, init_multinerd, pretrained_model, task):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    enc_ner = init_multinerd(
        tokenizer=tokenizer,
        task=task,
        system_prompt="system",
        filter_en=False,
        workers=0,
    ).train[0]

    enc_random = init_multinerd(
        tokenizer=tokenizer,
        task=task,
        system_prompt="random",
        filter_en=False,
        workers=0,
    ).train[0]

    enc_none = init_multinerd(
        tokenizer=tokenizer,
        task=task,
        system_prompt="none",
        filter_en=False,
        workers=0,
    ).train[0]
    return enc_ner, enc_none, enc_random, tokenizer


@app.cell
def _(enc_ner, tokenizer):
    print(tokenizer.decode(enc_ner["input_ids"]))
    print("\nLabel: ", enc_ner["labels"])
    return


@app.cell
def _(enc_random, tokenizer):
    print(tokenizer.decode(enc_random["input_ids"]))
    print("\nLabel: ", enc_random["labels"])
    return


@app.cell
def _(enc_none, tokenizer):
    print(tokenizer.decode(enc_none["input_ids"]))
    print("\nLabel: ", enc_none["labels"])
    return


if __name__ == "__main__":
    app.run()
