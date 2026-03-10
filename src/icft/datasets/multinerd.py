from collections.abc import Iterable
from typing import Literal, TypedDict, cast

import numpy as np
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from transformers import BatchEncoding, PreTrainedTokenizerFast

from icft.logging import logger
from icft.types import ICFTPrompt, ICFTTask

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

_id2label_full: dict[int, str] = {
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

id2label: dict[int, MultinerdTag] = {
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

label2id: dict[MultinerdTag, int] = {
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


class DatasetInfo(TypedDict):
    id2label: dict[int, str]
    label2id: dict[str, int]
    system_prompt: str


class MultinerdBatch(TypedDict):
    tokens: list[list[str]]
    ner_tags: list[list[MultinerdTag]]
    lang: list[MultinerdLang]


def _tokenize_seq_cls(
    batch: MultinerdBatch,
    tokenizer: PreTrainedTokenizerFast,
) -> BatchEncoding:
    prompts: list[str] = []
    labels: list[int] = []

    for tokens, tag_ids in zip(batch["tokens"], batch["ner_tags"]):
        sentence = " ".join(tokens)
        tokens, tag_ids = _join_spans(tokens=tokens, tag_ids=tag_ids)

        for token, tag_id in zip(tokens, tag_ids):
            prompt = _prompt_template(sentence=sentence, token=token)
            prompts.append(prompt)
            labels.append(tag_id)

    enc = tokenizer(prompts, add_special_tokens=False)
    enc["labels"] = labels

    return enc


def _tokenize_seq2seq(
    batch: MultinerdBatch,
    tokenizer: PreTrainedTokenizerFast,
) -> BatchEncoding:
    prompts: list[str] = []
    labels: list[str] = []

    for tokens, tag_ids in zip(batch["tokens"], batch["ner_tags"]):
        sentence = " ".join(tokens)
        tokens, tag_ids = _join_spans(tokens=tokens, tag_ids=tag_ids)

        for token, tag_id in zip(tokens, tag_ids):
            prompt = _prompt_template(sentence=sentence, token=token)
            prompts.append(prompt)
            labels.append(f"{id2label[tag_id]}{tokenizer.eos_token}")

    enc = tokenizer(prompts, add_special_tokens=False)
    labels_enc = tokenizer(labels, add_special_tokens=False)
    enc["labels"] = labels_enc["input_ids"]

    return enc


def _tokenize_causal_lm(
    batch: MultinerdBatch,
    tokenizer: PreTrainedTokenizerFast,
) -> BatchEncoding:
    ids: list[list[int]] = []
    attn: list[list[int]] = []
    labels: list[list[int]] = []

    for tokens, tag_ids in zip(batch["tokens"], batch["ner_tags"]):
        sentence = " ".join(tokens)
        tokens, tag_ids = _join_spans(tokens=tokens, tag_ids=tag_ids)

        for token, tag_id in zip(tokens, tag_ids):
            prompt = _prompt_template(sentence=sentence, token=token)
            answer = f"{id2label[tag_id]}{tokenizer.eos_token}"

            prompt_enc = tokenizer(prompt, add_special_tokens=False)
            answer_enc = tokenizer(answer, add_special_tokens=False)

            prompt_tokens = len(prompt_enc["input_ids"])

            _ids = prompt_enc["input_ids"] + answer_enc["input_ids"]
            _attn = prompt_enc["attention_mask"] + answer_enc["attention_mask"]
            _labels = [-100] * prompt_tokens + answer_enc["input_ids"]

            ids.append(_ids)
            attn.append(_attn)
            labels.append(_labels)

    enc = {"input_ids": ids, "attention_mask": attn, "labels": labels}
    return BatchEncoding(enc)


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


def _prepend_system_tokens(enc: BatchEncoding, sys: BatchEncoding) -> BatchEncoding:
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


def _join_spans(
    tokens: list[str],
    tag_ids: list[int],
) -> tuple[list[str], list[int]]:
    out_ids = []
    out_tokens = []
    for token, id in zip(tokens, tag_ids):
        tag = _id2label_full[id]

        if tag.startswith("B-"):
            tag = cast(MultinerdTag, tag[2:])
            out_ids.append(label2id[tag])
            out_tokens.append(token)
        elif tag.startswith("I-"):
            out_tokens[-1] = f"{out_tokens[-1]} {token}"
        else:
            tag = cast(MultinerdTag, tag)
            out_ids.append(label2id[tag])
            out_tokens.append(token)

    return out_tokens, out_ids


def init_system_prompt(
    tokenizer: PreTrainedTokenizerFast,
    task: ICFTTask,
    system_prompt: ICFTPrompt,
) -> BatchEncoding:
    cls_token = tokenizer.cls_token or "" if task == "seq-cls" else ""
    sys = tokenizer(f"{cls_token}{SYSTEM_PROMPT}", add_special_tokens=False)
    if system_prompt == "system":
        logger.debug("init system prompt")
    elif system_prompt == "random":
        logger.debug("randomize system prompt")
        sys = _randomize_system_prompt(tokenizer=tokenizer, sys=sys)
    elif system_prompt == "none":
        logger.debug("empty system prompt")
        ids = [tokenizer.cls_token_id] if cls_token else []
        attn = [1] if cls_token else []
        enc = {"input_ids": ids, "attention_mask": attn}
        sys = BatchEncoding(enc)
    else:
        raise NotImplementedError(f"System prompt '{system_prompt}'")

    logger.debug(
        "prepared system prompt with %d tokens",
        len(cast(list[int], sys["input_ids"])),
    )

    return sys


def _prompt_template(sentence: str, token: str) -> str:
    return f"Sentence: {sentence}\nTarget: {token}\nAnswer: "


def _filter_english(batch: MultinerdBatch) -> list[bool]:
    return [lang == "en" for lang in batch["lang"]]


def init_multinerd(
    tokenizer: PreTrainedTokenizerFast,
    task: ICFTTask,
    system_prompt: ICFTPrompt,
    filter_en: bool,
    workers: int,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    data = cast(DatasetDict, load_dataset("Babelscape/multinerd", split=split))

    if filter_en:
        logger.debug("filter multinerd english")
        data = data.filter(_filter_english, batched=True)

    logger.debug("tokenize multinerd for %s", task)
    if task == "seq2seq":
        tokenize_fn = _tokenize_seq2seq
    elif task == "seq-cls":
        tokenize_fn = _tokenize_seq_cls
    elif task == "causal-lm":
        tokenize_fn = _tokenize_causal_lm
    else:
        raise NotImplementedError(f"Task '{task}'")

    data = data.map(
        tokenize_fn,
        batched=True,
        fn_kwargs=dict(tokenizer=tokenizer),
        num_proc=workers,
        remove_columns=data["train"].column_names,
    )

    sys = init_system_prompt(
        tokenizer=tokenizer,
        task=task,
        system_prompt=system_prompt,
    )

    data = data.map(
        _prepend_system_tokens,
        batched=True,
        fn_kwargs=dict(sys=sys),
        num_proc=workers,
    )

    info = DatasetInfo(
        id2label=cast(dict[int, str], id2label),
        label2id=cast(dict[str, int], label2id),
        system_prompt=system_prompt,
    )

    return data, info
