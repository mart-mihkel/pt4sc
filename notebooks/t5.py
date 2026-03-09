import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():

    import marimo as mo
    import torch
    from torch import Tensor
    from torch.nn import CrossEntropyLoss
    from transformers import (
        AutoTokenizer,
        DataCollatorWithPadding,
        T5ForSequenceClassification,
    )

    from icft.datasets.multinerd import Multinerd

    return (
        AutoTokenizer,
        CrossEntropyLoss,
        DataCollatorWithPadding,
        Multinerd,
        T5ForSequenceClassification,
        Tensor,
        mo,
        torch,
    )


@app.cell
def _(
    AutoTokenizer,
    DataCollatorWithPadding,
    Multinerd,
    T5ForSequenceClassification,
):
    _model_path = "google-t5/t5-small"

    tokenizer = AutoTokenizer.from_pretrained(_model_path)

    _data = Multinerd(
        tokenizer=tokenizer,
        task="seq-cls",
        system_prompt="none",
        split=["train[:8]", "validation[:1]", "test[:1]"],
        filter_english=False,
    )

    _collator = DataCollatorWithPadding(tokenizer=tokenizer)
    batch = _collator(_data.train[:8])

    model = T5ForSequenceClassification.from_pretrained(
        _model_path,
        num_labels=len(Multinerd.ID2TAG),
        id2label=Multinerd.ID2TAG,
        label2id=Multinerd.TAG2ID,
    )

    encoder = model.transformer.encoder
    head = model.classification_head

    model
    return batch, encoder, head, model


@app.cell
def _(batch, model, torch):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    batch_size = input_ids.size(0)
    num_virtual_tokens = 10

    prefix = torch.randn(
        1,
        num_virtual_tokens,
        model.get_input_embeddings().embedding_dim,
    )
    return (
        attention_mask,
        batch_size,
        input_ids,
        labels,
        num_virtual_tokens,
        prefix,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Encoder-Decoder Forward Pass

    T5 is an encoder-decoder model, the input token ids for the decoder are [usually](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L614) taken as the encoder input ids prepended by the decoder start token. To deal with learnable embeddings we need to repeat the same on an embedding level.
    """)
    return


@app.cell
def _(T5ForSequenceClassification, Tensor, torch):
    def shift_right(model: T5ForSequenceClassification, inputs: Tensor) -> Tensor:
        decoder_start_token_id = model.config.decoder_start_token_id
        decoder_start_emb = model.get_input_embeddings().forward(
            torch.tensor(
                decoder_start_token_id,
                device=inputs.device,
            )
        )

        shifted_inputs = inputs.new_zeros(inputs.shape, device=inputs.device)
        shifted_inputs[..., 1:] = inputs[..., :-1].clone()
        shifted_inputs[:, ...] = decoder_start_emb

        return shifted_inputs

    return (shift_right,)


@app.cell
def _(
    attention_mask,
    batch_size,
    input_ids,
    model,
    num_virtual_tokens,
    prefix,
    shift_right,
    torch,
):
    base_emb = model.get_input_embeddings()
    input_emb = base_emb.forward(input_ids)
    prefix_emb = prefix.expand(batch_size, -1, -1)
    prefix_attn = torch.ones(
        batch_size,
        num_virtual_tokens,
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )

    inputs = torch.cat([prefix_emb, input_emb], dim=1)
    attn = torch.cat([prefix_attn, attention_mask], dim=1)
    decoder_inputs = shift_right(model, inputs)

    out = model.transformer(
        inputs_embeds=inputs,
        attention_mask=attn,
        decoder_inputs_embeds=decoder_inputs,
    )
    return attn, decoder_inputs, inputs, out


@app.cell
def _(CrossEntropyLoss, attn, head, labels, model, out):
    seq_out = out[0]
    sent_repr = (attn.unsqueeze(-1) * seq_out).mean(dim=1)

    logits = head(sent_repr)
    loss_fct = CrossEntropyLoss()
    loss_fct(
        logits.view(-1, model.config.num_labels),
        labels.to(logits.device).view(-1),
    )
    return (loss_fct,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Encoder Forward Pass

    Using just the encoder. For BERT models with a classification token prepended to the input sequence it is common to use first pooling, T5 has no such token so we use mean pooling.
    """)
    return


@app.cell
def _(attn, decoder_inputs, encoder, head, inputs, labels, loss_fct):
    enc_out = encoder(
        inputs_embeds=inputs,
        attention_mask=attn,
        decoder_inputs_embeds=decoder_inputs,
    )

    pooled_logits = enc_out.last_hidden_state.mean(dim=1)

    loss_fct(
        head(pooled_logits),
        labels.to(pooled_logits.device),
    )
    return


if __name__ == "__main__":
    app.run()
