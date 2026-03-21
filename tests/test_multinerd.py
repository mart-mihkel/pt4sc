from typing import cast

from datasets.splits import Split
from transformers import PreTrainedTokenizerFast

from icftsc.datasets.multinerd import _join_spans, init_multinerd

split = cast(
    Split,
    {
        "train": "train[:100]",
        "validation": "validation[:100]",
        "test": "test[:100]",
    },
)


def test_join_spans():
    tokens = ["Kuulus", "kohver", "Eston", "Kohver"]
    tag_ids = [0, 0, 1, 2]
    jtokens, jtags = _join_spans(tokens=tokens, tag_ids=tag_ids)

    assert jtokens == ["Kuulus", "kohver", "Eston Kohver"]
    assert jtags == [-1, -1, 0]


def test_multinerd_mmbert(mmbert_tokenizer: PreTrainedTokenizerFast):
    data, _ = init_multinerd(
        tokenizer=mmbert_tokenizer,
        model_type="modernbert",
        subset=1.0,
        filter_en=False,
        task="seqcls",
        split=split,
        workers=0,
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0


def test_multinerd_gpt2(gpt2_tokenizer):
    data, _ = init_multinerd(
        tokenizer=gpt2_tokenizer,
        model_type="gpt2",
        subset=1.0,
        filter_en=False,
        task="causal",
        split=split,
        workers=0,
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0


def test_multinerd_t5(t5_tokenizer):
    data, _ = init_multinerd(
        tokenizer=t5_tokenizer,
        model_type="t5",
        filter_en=False,
        subset=1.0,
        task="seq2seq",
        split=split,
        workers=0,
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0
