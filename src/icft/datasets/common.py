from collections.abc import Iterable
from typing import TypedDict, cast

import numpy as np
from transformers import BatchEncoding, PreTrainedTokenizerFast

from icft.logging import logger
from icft.types import PromptMode, Task


class DatasetInfo(TypedDict):
    id2label: dict[int, str]
    label2id: dict[str, int]
    system_prompt: str


def prepend_system_tokens(enc: BatchEncoding, sys: BatchEncoding) -> BatchEncoding:
    ids: list[list[int]] = []
    attn: list[list[int]] = []
    it = zip(
        cast(Iterable, enc["input_ids"]),
        cast(Iterable, enc["attention_mask"]),
        strict=True,
    )

    for _ids, _attn in it:
        ids.append(sys["input_ids"] + _ids)
        attn.append(sys["attention_mask"] + _attn)

    out = {"input_ids": ids, "attention_mask": attn, "labels": enc["labels"]}
    return BatchEncoding(out)


def init_system_prompt(
    tokenizer: PreTrainedTokenizerFast,
    task: Task,
    prompt_mode: PromptMode,
    system_prompt: str,
) -> BatchEncoding:
    cls_token = tokenizer.cls_token or "" if task == "seq-cls" else ""
    sys = tokenizer(f"{cls_token}{system_prompt}", add_special_tokens=False)
    if prompt_mode == "system":
        logger.debug("init system prompt")
    elif prompt_mode == "random":
        logger.debug("randomize system prompt")
        sys = _randomize_system_prompt(tokenizer=tokenizer, sys=sys)
    elif prompt_mode == "none":
        logger.debug("empty system prompt")
        ids = [tokenizer.cls_token_id] if cls_token else []
        attn = [1] if cls_token else []
        enc = {"input_ids": ids, "attention_mask": attn}
        sys = BatchEncoding(enc)
    else:
        raise NotImplementedError(f"System prompt '{prompt_mode}'")

    logger.debug(
        "prepared system prompt with %d tokens",
        len(cast(list[int], sys["input_ids"])),
    )

    return sys


def _randomize_system_prompt(
    tokenizer: PreTrainedTokenizerFast,
    sys: BatchEncoding,
) -> BatchEncoding:
    vocab_size = tokenizer.vocab_size
    special_ids = tokenizer.all_special_ids

    random_ids = []
    for token_id in cast(list[int], sys["input_ids"]):
        if token_id in special_ids:
            random_ids.append(token_id)
            continue

        rand_id = np.random.randint(0, vocab_size)
        while rand_id in special_ids:
            rand_id = np.random.randint(0, vocab_size)

        random_ids.append(rand_id)

    enc = {"input_ids": random_ids, "attention_mask": sys["attention_mask"]}
    return BatchEncoding(enc)
