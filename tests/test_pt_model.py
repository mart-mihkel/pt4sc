import tempfile

import torch
from torch.nn import Parameter
from transformers import (
    AutoConfig,
    DataCollatorWithPadding,
    PreTrainedTokenizerFast,
)

from icftsc.datasets.multinerd import DatasetInfo
from icftsc.modeling.common import PTModelConfig
from icftsc.modeling.seqcls import (
    PTBertForSequenceClassification,
    PTGPTForSequenceClassification,
    PTT5ForSequenceClassification,
)
from icftsc.scripts.prompt_tune import init_pt_model


def test_init_pt_bert(mmbert_tokenizer: PreTrainedTokenizerFast):
    info = DatasetInfo(
        id2label={0: "0", 1: "1"},
        label2id={"0": 0, "1": 1},
        system_prompt="mock",
    )

    model = init_pt_model(
        model_path="jhu-clsp/mmBERT-base",
        tokenizer=mmbert_tokenizer,
        prefix_init="pretrained",
        data_info=info,
        task="seqcls",
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

    collate_fn = DataCollatorWithPadding(
        tokenizer=mmbert_tokenizer,
        pad_to_multiple_of=8,
    )

    config = PTModelConfig(
        pretrained_model="jhu-clsp/mmBERT-base",
        num_virtual_tokens=10,
        task="seqcls",
        num_labels=2,
        id2label={0: "0", 1: "1"},
        label2id={"0": 0, "1": 1},
    )

    model = PTBertForSequenceClassification(config=config)
    out = model(**collate_fn(data))

    assert out.loss is not None
    assert out.logits is not None
    assert out.logits.shape == (2, 2)


def test_pt_gpt2_seq_cls(gpt2_tokenizer: PreTrainedTokenizerFast):
    data = [
        {"input_ids": [1, 2], "label": 0},
        {"input_ids": [3], "label": 1},
    ]

    collate_fn = DataCollatorWithPadding(
        tokenizer=gpt2_tokenizer,
        pad_to_multiple_of=8,
    )

    config = PTModelConfig(
        pretrained_model="openai-community/gpt2",
        num_virtual_tokens=10,
        task="seqcls",
        num_labels=2,
        id2label={0: "0", 1: "1"},
        label2id={"0": 0, "1": 1},
    )

    model = PTGPTForSequenceClassification(config=config)
    model.base.config.pad_token_id = gpt2_tokenizer.eos_token_id
    out = model(**collate_fn(data))

    assert out.loss is not None
    assert out.logits is not None
    assert out.logits.shape == (2, 2)


def test_pt_t5(t5_tokenizer: PreTrainedTokenizerFast):
    data = [
        {"input_ids": [1, 2], "label": 0},
        {"input_ids": [3], "label": 1},
    ]

    collate_fn = DataCollatorWithPadding(
        tokenizer=t5_tokenizer,
        pad_to_multiple_of=8,
    )

    config = PTModelConfig(
        pretrained_model="google-t5/t5-small",
        num_virtual_tokens=10,
        task="seqcls",
        num_labels=2,
        id2label={0: "0", 1: "1"},
        label2id={"0": 0, "1": 1},
    )

    model = PTT5ForSequenceClassification(config=config)
    model.base.config.pad_token_id = t5_tokenizer.eos_token_id
    out = model(**collate_fn(data))

    assert out.loss is not None
    assert out.logits is not None
    assert out.logits.shape == (2, 2)


def test_save_model_seqcls(mmbert_tokenizer: PreTrainedTokenizerFast):
    info = DatasetInfo(
        id2label={0: "0", 1: "1"},
        label2id={"0": 0, "1": 1},
        system_prompt="mock",
    )

    model = init_pt_model(
        model_path="jhu-clsp/mmBERT-base",
        tokenizer=mmbert_tokenizer,
        prefix_init="pretrained",
        data_info=info,
        task="seqcls",
    )

    new_prefix = Parameter(torch.ones_like(model.prefix))
    with tempfile.TemporaryDirectory() as tmp:
        model.prefix = new_prefix
        model.save_pretrained(tmp)

        config = AutoConfig.from_pretrained(tmp)
        loaded_model = PTBertForSequenceClassification.from_pretrained(
            tmp,
            config=config,
        )

    assert isinstance(loaded_model, PTBertForSequenceClassification)
    assert loaded_model.prefix.equal(new_prefix)
