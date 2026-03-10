from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from icft.datasets.multinerd import init_multinerd


def test_multinerd_mmbert():
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base"),
    )

    data, _ = init_multinerd(
        tokenizer=tokenizer,
        task="seq-cls",
        system_prompt="system",
        filter_en=False,
        workers=0,
        split={"train": "train[:1]"},  # type: ignore
    )

    assert len(data["train"]) > 0


def test_multinerd_gpt2():
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("openai-community/gpt2"),
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    data, _ = init_multinerd(
        tokenizer=tokenizer,
        task="causal-lm",
        system_prompt="system",
        filter_en=False,
        workers=0,
        split={"train": "train[:1]"},  # type: ignore
    )

    assert len(data["train"]) > 0


def test_multinerd_t5():
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("google-t5/t5-small"),
    )

    data, _ = init_multinerd(
        tokenizer=tokenizer,
        task="seq2seq",
        system_prompt="system",
        filter_en=False,
        workers=0,
        split={"train": "train[:1]"},  # type: ignore
    )

    assert len(data["train"]) > 0
