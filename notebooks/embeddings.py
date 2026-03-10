import marimo

__generated_with = "0.20.1"
app = marimo.App()


@app.cell
def _():
    from typing import cast

    import marimo as mo
    from torch import Tensor
    from transformers import (
        AutoModel,
        AutoTokenizer,
        BertModel,
        PreTrainedTokenizer,
    )

    from icft.datasets.multinerd import SYSTEM_PROMPT

    return (
        AutoModel,
        AutoTokenizer,
        BertModel,
        SYSTEM_PROMPT,
        PreTrainedTokenizer,
        Tensor,
        cast,
        mo,
    )


@app.cell
def _(BertModel, PreTrainedTokenizer, Tensor):
    def encode_prefix(
        prefix: str,
        model: BertModel,
        tokenizer: PreTrainedTokenizer,
    ) -> Tensor:
        prefix_tokenized = tokenizer(prefix, return_tensors="pt")
        token_ids = prefix_tokenized["input_ids"][0]
        return model.get_input_embeddings().forward(token_ids)

    return (encode_prefix,)


@app.cell
def _(BertModel, PreTrainedTokenizer, Tensor, cast):
    def decode_prefix(
        prefix_embeddings: Tensor,
        model: BertModel,
        tokenizer: PreTrainedTokenizer,
    ) -> str | list[str]:
        voc_embeddings = cast(Tensor, model.get_input_embeddings().weight)
        similarity = prefix_embeddings @ voc_embeddings.T
        token_ids = similarity.argmax(dim=1)
        return tokenizer.decode(token_ids=token_ids)

    return (decode_prefix,)


@app.cell
def _(AutoModel, AutoTokenizer):
    _pretrained_model = "jhu-clsp/mmBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(_pretrained_model)
    model = AutoModel.from_pretrained(_pretrained_model)
    return model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Decode mmBERT system prompt embeddings using cosine similarity
    """)
    return


@app.cell
def _(SYSTEM_PROMPT, decode_prefix, encode_prefix, model, tokenizer):
    _prefix_embeddings = encode_prefix(
        prefix=SYSTEM_PROMPT,
        model=model,
        tokenizer=tokenizer,
    )

    _prefix_decoded = decode_prefix(
        prefix_embeddings=_prefix_embeddings,
        model=model,
        tokenizer=tokenizer,
    )

    print(_prefix_decoded)
    return


if __name__ == "__main__":
    app.run()
