from transformers import PreTrainedTokenizerFast

from icft.common import init_collate_fn
from icft.datasets.multinerd import DatasetInfo
from icft.models import (
    PTDecoderModel,
    PTDecoderModelConfig,
    PTEncoderDecoderModel,
    PTEncoderDecoderModelConfig,
    PTEncoderModel,
    PTEncoderModelConfig,
)
from icft.scripts.prompt_tune import init_pt_model


def test_init_pt_bert(mmbert_tokenizer: PreTrainedTokenizerFast):
    info = DatasetInfo(
        id2label={0: "0", 1: "1"},
        label2id={"0": 0, "1": 1},
        system_prompt="mock",
    )

    model = init_pt_model(
        task="seq-cls",
        prefix_init="pretrained",
        tokenizer=mmbert_tokenizer,
        model_path="jhu-clsp/mmBERT-base",
        data_info=info,
    )

    assert model is not None
    assert model.base is not None
    assert model.prefix is not None


def test_pt_bert(mmbert_tokenizer: PreTrainedTokenizerFast):
    cls = mmbert_tokenizer.cls_token_id
    data = [
        {"input_ids": [cls, 1, 2], "label": 0},
        {"input_ids": [cls, 3], "label": 1},
    ]

    config = PTEncoderModelConfig(
        task="seq-cls",
        pretrained_model="jhu-clsp/mmBERT-base",
        num_virtual_tokens=10,
        num_labels=2,
        id2label={0: "0", 1: "1"},
        label2id={"0": 0, "1": 1},
    )

    model = PTEncoderModel(config=config)
    collate_fn = init_collate_fn(tokenizer=mmbert_tokenizer, task="seq-cls")
    out = model(**collate_fn(data))

    assert out.loss is not None
    assert out.logits is not None
    assert out.logits.shape == (2, 2)


def test_pt_gpt2_causal_lm(gpt2_tokenizer: PreTrainedTokenizerFast):
    eos = gpt2_tokenizer.eos_token_id
    data = [
        {"input_ids": [1, 2, 3, eos], "labels": [-100, -100, 3, eos]},
        {"input_ids": [3, 4, eos], "labels": [-100, 4, eos]},
    ]

    config = PTDecoderModelConfig(
        task="causal-lm",
        pretrained_model="openai-community/gpt2",
        num_virtual_tokens=10,
        num_labels=2,
    )

    model = PTDecoderModel(config=config)
    model.base.config.pad_token_id = gpt2_tokenizer.eos_token_id
    collate_fn = init_collate_fn(tokenizer=gpt2_tokenizer, task="causal-lm")
    out = model(**collate_fn(data))

    assert out.loss is not None
    assert out.logits is not None


def test_pt_gpt2_seq_cls(gpt2_tokenizer: PreTrainedTokenizerFast):
    data = [
        {"input_ids": [1, 2], "label": 0},
        {"input_ids": [3], "label": 1},
    ]

    config = PTDecoderModelConfig(
        task="seq-cls",
        pretrained_model="openai-community/gpt2",
        num_virtual_tokens=10,
        num_labels=2,
        id2label={0: "0", 1: "1"},
        label2id={"0": 0, "1": 1},
    )

    model = PTDecoderModel(config=config)
    model.base.config.pad_token_id = gpt2_tokenizer.eos_token_id
    collate_fn = init_collate_fn(tokenizer=gpt2_tokenizer, task="seq-cls")
    out = model(**collate_fn(data))

    assert out.loss is not None
    assert out.logits is not None
    assert out.logits.shape == (2, 2)


def test_pt_t5_seq2seq(t5_tokenizer: PreTrainedTokenizerFast):
    eos = t5_tokenizer.eos_token_id
    data = [
        {"input_ids": [34, 231, eos], "labels": [453, eos]},
        {"input_ids": [123, eos], "labels": [64, eos]},
    ]

    config = PTEncoderDecoderModelConfig(
        task="seq2seq",
        pretrained_model="google-t5/t5-small",
        num_virtual_tokens=10,
    )

    model = PTEncoderDecoderModel(config=config)
    model.base.config.pad_token_id = t5_tokenizer.eos_token_id
    collate_fn = init_collate_fn(tokenizer=t5_tokenizer, task="seq2seq")
    out = model(**collate_fn(data))

    assert out.loss is not None
    assert out.logits is not None


def test_pt_t5_seq_cls(t5_tokenizer: PreTrainedTokenizerFast):
    data = [
        {"input_ids": [1, 2], "label": 0},
        {"input_ids": [3], "label": 1},
    ]

    config = PTEncoderDecoderModelConfig(
        task="seq-cls",
        pretrained_model="google-t5/t5-small",
        num_virtual_tokens=10,
        num_labels=2,
        id2label={0: "0", 1: "1"},
        label2id={"0": 0, "1": 1},
    )

    model = PTEncoderDecoderModel(config=config)
    model.base.config.pad_token_id = t5_tokenizer.eos_token_id
    collate_fn = init_collate_fn(tokenizer=t5_tokenizer, task="seq-cls")
    out = model(**collate_fn(data))

    assert out.loss is not None
    assert out.logits is not None
    assert out.logits.shape == (2, 2)
