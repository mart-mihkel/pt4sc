from typing import cast

from pytest import fixture
from transformers import AutoTokenizer, PreTrainedTokenizerFast


@fixture
def mmbert_tokenizer() -> PreTrainedTokenizerFast:
    return cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base"),
    )


@fixture
def gpt2_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("openai-community/gpt2"),
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


@fixture
def t5_tokenizer() -> PreTrainedTokenizerFast:
    return cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("google-t5/t5-small"),
    )
