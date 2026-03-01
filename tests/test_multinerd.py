from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from icft.datasets.multinerd import Multinerd


def test_join_spans():
    tokens = ["New", "York", "is", "sus", "."]
    ids = [5, 4, 0, 0, 0]

    tokens, ids = Multinerd._join_spans(tokens, ids)

    assert tokens == ["New York", "is", "sus", "."]
    assert ids == [3, 0, 0, 0]


def test_multinerd_mmbert():
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base"),
    )

    data = Multinerd(
        tokenizer=tokenizer,
        task="seq-cls",
        system_prompt_mode="ner",
        split=["train[:10]", "validation[:10]", "test[:10]"],
        filter_english=False,
    )

    assert len(data.train) > 0
    assert len(data.eval) > 0
    assert len(data.test) > 0


def test_multinerd_gpt2():
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("openai-community/gpt2"),
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    data = Multinerd(
        tokenizer=tokenizer,
        task="causal-lm",
        system_prompt_mode="ner",
        split=["train[:10]", "validation[:10]", "test[:10]"],
        filter_english=False,
    )

    assert len(data.train) > 0
    assert len(data.eval) > 0
    assert len(data.test) > 0


def test_multinerd_t5():
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("google-t5/t5-small"),
    )

    data = Multinerd(
        tokenizer=tokenizer,
        task="seq2seq",
        system_prompt_mode="ner",
        split=["train[:10]", "validation[:10]", "test[:10]"],
        filter_english=False,
    )

    assert len(data.train) > 0
    assert len(data.eval) > 0
    assert len(data.test) > 0
