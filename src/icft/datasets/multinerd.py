from typing import Iterable, Literal, TypedDict, cast

import numpy as np
from datasets.load import load_dataset
from datasets.utils.info_utils import VerificationMode
from transformers import BatchEncoding, PreTrainedTokenizerFast

from icft.logging import logger
from icft.types import ICFTTask, PromptMode

type MultinerdLang = Literal[
    "zh",
    "nl",
    "en",
    "fr",
    "de",
    "it",
    "pl",
    "pt",
    "ru",
    "es",
]

type MultinerdTag = Literal[
    "O",
    "PER",
    "ORG",
    "LOC",
    "ANIM",
    "BIO",
    "CEL",
    "DIS",
    "EVE",
    "FOOD",
    "INST",
    "MEDIA",
    "MYTH",
    "PLANT",
    "TIME",
    "VEHI",
]


class MultinerdBatch(TypedDict):
    tokens: list[list[str]]
    ner_tags: list[list[MultinerdTag]]
    lang: list[MultinerdLang]


class Multinerd:
    _ID2TAG_FULL: dict[int, str] = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-ANIM",
        8: "I-ANIM",
        9: "B-BIO",
        10: "I-BIO",
        11: "B-CEL",
        12: "I-CEL",
        13: "B-DIS",
        14: "I-DIS",
        15: "B-EVE",
        16: "I-EVE",
        17: "B-FOOD",
        18: "I-FOOD",
        19: "B-INST",
        20: "I-INST",
        21: "B-MEDIA",
        22: "I-MEDIA",
        23: "B-MYTH",
        24: "I-MYTH",
        25: "B-PLANT",
        26: "I-PLANT",
        27: "B-TIME",
        28: "I-TIME",
        29: "B-VEHI",
        30: "I-VEHI",
    }

    ID2TAG: dict[int, MultinerdTag] = {
        0: "O",
        1: "PER",
        2: "ORG",
        3: "LOC",
        4: "ANIM",
        5: "BIO",
        6: "CEL",
        7: "DIS",
        8: "EVE",
        9: "FOOD",
        10: "INST",
        11: "MEDIA",
        12: "MYTH",
        13: "PLANT",
        14: "TIME",
        15: "VEHI",
    }

    TAG2ID: dict[MultinerdTag, int] = {
        "O": 0,
        "PER": 1,
        "ORG": 2,
        "LOC": 3,
        "ANIM": 4,
        "BIO": 5,
        "CEL": 6,
        "DIS": 7,
        "EVE": 8,
        "FOOD": 9,
        "INST": 10,
        "MEDIA": 11,
        "MYTH": 12,
        "PLANT": 13,
        "TIME": 14,
        "VEHI": 15,
    }

    SYSTEM_PROMPT = """Task: Named Entity Recognition

Classify the NER tag of the target entity in the sentence:

Sentence: Paris is the capital of France.
Target: Paris
Answer: LOC

"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        task: ICFTTask,
        system_prompt_mode: PromptMode = "none",
        split: list[str] = ["train", "validation", "test"],
        workers: int = 0,
        filter_english: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.system_prompt_mode = system_prompt_mode

        cls_token = tokenizer.cls_token if task == "seq-cls" else ""
        self.system_tokens = self.tokenizer(
            f"{cls_token}{self.SYSTEM_PROMPT}",
            add_special_tokens=False,
        )

        self.none_tokens = BatchEncoding(
            {
                "input_ids": [tokenizer.cls_token_id] if cls_token else [],
                "attention_mask": [1] if cls_token else [],
            }
        )

        self.random_tokens = self._randomize_system_prompt()

        logger.debug(
            "prepared multinerd system prompt with %d tokens",
            len(self.system_tokens["input_ids"]),
        )

        train, eval, test = load_dataset(
            "Babelscape/multinerd",
            split=split,
            verification_mode=VerificationMode.NO_CHECKS,
        )

        if filter_english:
            logger.debug("filter multinerd english")
            train = train.filter(self._filter_english, batched=True)
            eval = eval.filter(self._filter_english, batched=True)
            test = test.filter(self._filter_english, batched=True)

        if task == "seq2seq":
            tokenize_fn = self._tokenize_seq2seq
        elif task == "seq-cls":
            tokenize_fn = self._tokenize_seq_cls
        elif task == "causal-lm":
            tokenize_fn = self._tokenize_causal_lm
        else:
            raise NotImplementedError(f"Task '{task}'")

        logger.debug("tokenize multinerd for %s", task)

        self.train = train.map(
            tokenize_fn,
            batched=True,
            remove_columns=train.column_names,
            num_proc=workers,
        )

        self.eval = eval.map(
            tokenize_fn,
            batched=True,
            remove_columns=eval.column_names,
            num_proc=workers,
        )

        self.test = test.map(
            tokenize_fn,
            batched=True,
            remove_columns=test.column_names,
            num_proc=workers,
        )

    def _randomize_system_prompt(self) -> BatchEncoding:
        vocab_size = self.tokenizer.vocab_size
        special_ids = self.tokenizer.all_special_ids

        random_ids = []
        for token_id in self.system_tokens["input_ids"]:
            if token_id in special_ids:
                random_ids.append(token_id)
                continue

            rand_id = np.random.randint(0, vocab_size)
            while rand_id in special_ids:
                rand_id = np.random.randint(0, vocab_size)

            random_ids.append(rand_id)

        return BatchEncoding(
            {
                "input_ids": random_ids,
                "attention_mask": self.system_tokens["attention_mask"],
            }
        )

    def _tokenize_seq_cls(self, batch: MultinerdBatch) -> BatchEncoding:
        prompts: list[str] = []
        labels: list[int] = []

        for tokens, tag_ids in zip(batch["tokens"], batch["ner_tags"]):
            sentence = " ".join(tokens)
            tokens, tag_ids = self._join_spans(tokens=tokens, tag_ids=tag_ids)

            for token, tag_id in zip(tokens, tag_ids):
                prompt = self._prompt_template(sentence=sentence, token=token)
                prompts.append(prompt)
                labels.append(tag_id)

        enc = self.tokenizer(prompts, add_special_tokens=False)
        enc["labels"] = labels

        return self._maybe_prepend_system_tokens(enc=enc)

    def _tokenize_seq2seq(self, batch: MultinerdBatch) -> BatchEncoding:
        prompts: list[str] = []
        labels: list[str] = []

        for tokens, tag_ids in zip(batch["tokens"], batch["ner_tags"]):
            sentence = " ".join(tokens)
            tokens, tag_ids = self._join_spans(tokens=tokens, tag_ids=tag_ids)

            for token, tag_id in zip(tokens, tag_ids):
                prompt = self._prompt_template(sentence=sentence, token=token)
                prompts.append(prompt)
                labels.append(f"{self.ID2TAG[tag_id]}{self.tokenizer.eos_token}")

        prompts_enc = self.tokenizer(prompts, add_special_tokens=False)
        labels_enc = self.tokenizer(labels, add_special_tokens=False)
        prompts_enc["labels"] = labels_enc["input_ids"]

        return self._maybe_prepend_system_tokens(enc=prompts_enc)

    def _tokenize_causal_lm(self, batch: MultinerdBatch) -> BatchEncoding:
        ids: list[list[int]] = []
        attn: list[list[int]] = []
        labels: list[list[int]] = []

        for tokens, tag_ids in zip(batch["tokens"], batch["ner_tags"]):
            sentence = " ".join(tokens)
            tokens, tag_ids = self._join_spans(tokens=tokens, tag_ids=tag_ids)

            for token, tag_id in zip(tokens, tag_ids):
                prompt = self._prompt_template(sentence=sentence, token=token)
                answer = f"{self.ID2TAG[tag_id]}{self.tokenizer.eos_token}"

                prompt_enc = self.tokenizer(prompt, add_special_tokens=False)
                answer_enc = self.tokenizer(answer, add_special_tokens=False)

                prompt_tokens = len(prompt_enc["input_ids"])

                _ids = prompt_enc["input_ids"] + answer_enc["input_ids"]
                _attn = prompt_enc["attention_mask"] + answer_enc["attention_mask"]
                _labels = [-100] * prompt_tokens + answer_enc["input_ids"]

                ids.append(_ids)
                attn.append(_attn)
                labels.append(_labels)

        enc = BatchEncoding(
            {
                "input_ids": ids,
                "attention_mask": attn,
                "labels": labels,
            }
        )

        return self._maybe_prepend_system_tokens(enc=enc)

    def _maybe_prepend_system_tokens(
        self,
        enc: BatchEncoding,
    ) -> BatchEncoding:
        if self.system_prompt_mode == "none":
            sys_tokens = self.none_tokens
        elif self.system_prompt_mode == "ner":
            sys_tokens = self.system_tokens
        elif self.system_prompt_mode == "random":
            sys_tokens = self.random_tokens
        else:
            raise NotImplementedError(f"System prompt mode '{self.system_prompt_mode}'")

        ids: list[list[int]] = []
        attn: list[list[int]] = []
        it = zip(
            cast(Iterable, enc["input_ids"]),
            cast(Iterable, enc["attention_mask"]),
        )

        for _ids, _attn in it:
            ids.append(sys_tokens["input_ids"] + _ids)
            attn.append(sys_tokens["attention_mask"] + _attn)

        return BatchEncoding(
            {"input_ids": ids, "attention_mask": attn, "labels": enc["labels"]}
        )

    @staticmethod
    def _prompt_template(sentence: str, token: str) -> str:
        return f"Sentence: {sentence}\nTarget: {token}\nAnswer: "

    @staticmethod
    def _join_spans(
        tokens: list[str],
        tag_ids: list[int],
    ) -> tuple[list[str], list[int]]:
        out_ids = []
        out_tokens = []
        for token, id in zip(tokens, tag_ids):
            tag = Multinerd._ID2TAG_FULL[id]

            if tag.startswith("B-"):
                tag = cast(MultinerdTag, tag[2:])
                out_ids.append(Multinerd.TAG2ID[tag])
                out_tokens.append(token)
            elif tag.startswith("I-"):
                out_tokens[-1] = f"{out_tokens[-1]} {token}"
            else:
                tag = cast(MultinerdTag, tag)
                out_ids.append(Multinerd.TAG2ID[tag])
                out_tokens.append(token)

        return out_tokens, out_ids

    @staticmethod
    def _filter_english(batch: MultinerdBatch) -> list[bool]:
        return [lang == "en" for lang in batch["lang"]]
