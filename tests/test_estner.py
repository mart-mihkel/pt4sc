from typing import cast

from datasets.splits import Split
from transformers import PreTrainedTokenizerFast

from icft.datasets.estner import init_estner

split = cast(
    Split,
    {
        "train": "train[:100]",
        "dev": "dev[:100]",
        "test": "test[:100]",
    },
)


def test_estner_mmbert(mmbert_tokenizer: PreTrainedTokenizerFast):
    data, _ = init_estner(
        tokenizer=mmbert_tokenizer,
        task="seq-cls",
        prompt_mode="system",
        workers=0,
        split=split,
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0


def test_estner_gpt2(gpt2_tokenizer: PreTrainedTokenizerFast):
    data, _ = init_estner(
        tokenizer=gpt2_tokenizer,
        task="causal-lm",
        prompt_mode="system",
        workers=0,
        split=split,
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0


def test_estner_t5(t5_tokenizer: PreTrainedTokenizerFast):
    data, _ = init_estner(
        tokenizer=t5_tokenizer,
        task="seq2seq",
        prompt_mode="system",
        workers=0,
        split=split,
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0
