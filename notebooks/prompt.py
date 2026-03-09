import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    from transformers import AutoTokenizer

    from icft.datasets.multinerd import Multinerd

    return AutoTokenizer, Multinerd


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
def _(AutoTokenizer, Multinerd, pretrained_model, task):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    enc_ner = Multinerd(
        tokenizer=tokenizer,
        task=task,
        system_prompt="system",
        split=["train[:1]", "validation[:1]", "test[:1]"],
        filter_english=False,
    ).train[0]

    enc_random = Multinerd(
        tokenizer=tokenizer,
        task=task,
        system_prompt="random",
        split=["train[:1]", "validation[:1]", "test[:1]"],
        filter_english=False,
    ).train[0]

    enc_none = Multinerd(
        tokenizer=tokenizer,
        task=task,
        system_prompt="none",
        split=["train[:1]", "validation[:1]", "test[:1]"],
        filter_english=False,
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
