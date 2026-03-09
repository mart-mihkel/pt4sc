from typing import Iterable, cast

import numpy as np
import datasets
from transformers import BatchEncoding, PreTrainedTokenizerFast

from icft.types import ICFTPrompt, ICFTTask
from icft.logging import logger


class Dataset:
    SYSTEM_PROMPT: str = ""
    ID2LABEL: dict[int, str] = {}
    LABEL2ID: dict[str, int] = {}

    train: datasets.Dataset
    eval: datasets.Dataset
    test: datasets.Dataset

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        task: ICFTTask,
        system_prompt: ICFTPrompt = "none",
    ) -> None:
        cls_token = tokenizer.cls_token or "" if task == "seq-cls" else ""

        self.tokenizer = tokenizer
        self.task = task
        self.system_prompt_tokens = self.tokenizer(
            f"{cls_token}{self.SYSTEM_PROMPT}",
            add_special_tokens=False,
        )

        if system_prompt == "system":
            logger.debug("init system prompt")
            self.system_tokens = self.system_prompt_tokens
        elif system_prompt == "random":
            logger.debug("randomize system prompt")
            self.system_tokens = self._randomize_system_prompt()
        elif system_prompt == "none":
            logger.debug("empty system prompt")
            ids = [tokenizer.cls_token_id] if cls_token else []
            attn = [1] if cls_token else []
            enc = {"input_ids": ids, "attention_mask": attn}
            self.system_tokens = BatchEncoding(enc)
        else:
            raise NotImplementedError(f"System prompt '{system_prompt}'")

        logger.debug(
            "prepared prompt with %d tokens",
            len(cast(list[int], self.system_tokens["input_ids"])),
        )

    def _randomize_system_prompt(self) -> BatchEncoding:
        vocab_size = self.tokenizer.vocab_size
        special_ids = self.tokenizer.all_special_ids

        random_ids = []
        for token_id in self.system_prompt_tokens["input_ids"]:
            if token_id in special_ids:
                random_ids.append(token_id)
                continue

            rand_id = np.random.randint(0, vocab_size)
            while rand_id in special_ids:
                rand_id = np.random.randint(0, vocab_size)

            random_ids.append(rand_id)

        enc = {
            "input_ids": random_ids,
            "attention_mask": self.system_prompt_tokens["attention_mask"],
        }

        return BatchEncoding(enc)

    @staticmethod
    def _prepend_system_tokens(
        enc: BatchEncoding,
        sys: BatchEncoding,
    ) -> BatchEncoding:
        ids: list[list[int]] = []
        attn: list[list[int]] = []
        it = zip(
            cast(Iterable, enc["input_ids"]),
            cast(Iterable, enc["attention_mask"]),
        )

        for _ids, _attn in it:
            ids.append(sys["input_ids"] + _ids)
            attn.append(sys["attention_mask"] + _attn)

        out = {"input_ids": ids, "attention_mask": attn, "labels": enc["labels"]}
        return BatchEncoding(out)
