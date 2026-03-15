from typing import Literal, TypedDict, cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from transformers import BatchEncoding, PreTrainedTokenizerFast

from icft.datasets.common import DatasetInfo, init_system_prompt, prepend_system_tokens
from icft.logging import logger
from icft.types import PromptMode, Task

type EstnerTag = Literal[
    "O",
    "PER",
    "GPE",
    "LOC",
    "ORG",
    "PROD",
    "EVENT",
    "DATE",
    "TIME",
    "TITLE",
    "MONEY",
    "PERCENT",
]

id2label: dict[int, EstnerTag] = {
    0: "O",
    1: "PER",
    2: "GPE",
    3: "LOC",
    4: "ORG",
    5: "PROD",
    6: "EVENT",
    7: "DATE",
    8: "TIME",
    9: "TITLE",
    10: "MONEY",
    11: "PERCENT",
}

label2id: dict[EstnerTag, int] = {
    "O": 0,
    "PER": 1,
    "GPE": 2,
    "LOC": 3,
    "ORG": 4,
    "PROD": 5,
    "EVENT": 6,
    "DATE": 7,
    "TIME": 8,
    "TITLE": 9,
    "MONEY": 10,
    "PERCENT": 11,
}


system_prompt = """Ülesanne: Nimeüksuste tuvastamine (NER)

Määra sõna NER-märgend antud lauses.

Lause: Pariis on Prantusmaa pealinn.
Sõna: Pariis
Vastus: LOC

"""


class EstnerBatch(TypedDict):
    doc_id: list[int]
    sent_id: list[int]
    tokens: list[list[str]]
    ner_tags: list[list[str]]
    ner_tags1: list[list[str]]
    ner_tags2: list[list[str]]


def _tokenize_seq_cls(
    batch: EstnerBatch,
    tokenizer: PreTrainedTokenizerFast,
) -> BatchEncoding:
    prompts: list[str] = []
    labels: list[int] = []

    for tokens, tags in zip(batch["tokens"], batch["ner_tags"], strict=True):
        sentence = " ".join(tokens)
        tokens, tags = _join_spans(tokens=tokens, tags=tags)

        for token, tag in zip(tokens, tags, strict=True):
            prompt = _prompt_template(sentence=sentence, token=token)
            prompts.append(prompt)
            labels.append(label2id[tag])

    enc = tokenizer(prompts, add_special_tokens=False, truncation=True)
    enc["labels"] = labels

    return enc


def _tokenize_seq2seq(
    batch: EstnerBatch,
    tokenizer: PreTrainedTokenizerFast,
) -> BatchEncoding:
    prompts: list[str] = []
    labels: list[str] = []

    for tokens, tags in zip(batch["tokens"], batch["ner_tags"], strict=True):
        sentence = " ".join(tokens)
        tokens, tags = _join_spans(tokens=tokens, tags=tags)

        for token, tag in zip(tokens, tags, strict=True):
            prompt = _prompt_template(sentence=sentence, token=token)
            prompts.append(prompt)
            labels.append(f"{tag}{tokenizer.eos_token}")

    enc = tokenizer(prompts, add_special_tokens=False, truncation=True)
    labels_enc = tokenizer(labels, add_special_tokens=False, truncation=True)
    enc["labels"] = labels_enc["input_ids"]

    return enc


def _tokenize_causal_lm(
    batch: EstnerBatch,
    tokenizer: PreTrainedTokenizerFast,
) -> BatchEncoding:
    ids: list[list[int]] = []
    attn: list[list[int]] = []
    labels: list[list[int]] = []

    for tokens, tags in zip(batch["tokens"], batch["ner_tags"], strict=True):
        sentence = " ".join(tokens)
        tokens, tags = _join_spans(tokens=tokens, tags=tags)

        for token, tag in zip(tokens, tags, strict=True):
            prompt = _prompt_template(sentence=sentence, token=token)
            answer = f"{tag}{tokenizer.eos_token}"

            prompt_enc = tokenizer(prompt, add_special_tokens=False, truncation=True)
            answer_enc = tokenizer(answer, add_special_tokens=False, truncation=True)

            prompt_tokens = len(prompt_enc["input_ids"])

            _ids = prompt_enc["input_ids"] + answer_enc["input_ids"]
            _attn = prompt_enc["attention_mask"] + answer_enc["attention_mask"]
            _labels = [-100] * prompt_tokens + answer_enc["input_ids"]

            ids.append(_ids)
            attn.append(_attn)
            labels.append(_labels)

    enc = {"input_ids": ids, "attention_mask": attn, "labels": labels}
    return BatchEncoding(enc)


def _join_spans(
    tokens: list[str],
    tags: list[str],
) -> tuple[list[str], list[EstnerTag]]:
    out_tags = []
    out_tokens = []
    for token, tag in zip(tokens, tags, strict=True):
        if tag.startswith("B-"):
            tag = cast(EstnerTag, tag[2:])
            out_tags.append(tag)
            out_tokens.append(token)
        elif tag.startswith("I-"):
            out_tokens[-1] = f"{out_tokens[-1]} {token}"
        else:
            tag = cast(EstnerTag, tag)
            out_tags.append(tag)
            out_tokens.append(token)

    return out_tokens, out_tags


def _prompt_template(sentence: str, token: str) -> str:
    return f"Lause: {sentence}\nSõna: {token}\nVastus: "


def init_estner(
    tokenizer: PreTrainedTokenizerFast,
    task: Task,
    prompt_mode: PromptMode,
    workers: int,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    data = cast(DatasetDict, load_dataset("tartuNLP/EstNER", split=split))

    logger.debug("tokenize estner for %s", task)
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
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=workers,
        remove_columns=next(iter(data.values())).column_names,
    )

    sys = init_system_prompt(
        tokenizer=tokenizer,
        task=task,
        prompt_mode=prompt_mode,
        system_prompt=system_prompt,
    )

    if len(cast(list[int], sys["input_ids"])) > 0:
        logger.debug("prepend system tokens")
        data = data.map(
            prepend_system_tokens,
            batched=True,
            fn_kwargs={"sys": sys},
            num_proc=workers,
        )

    info = DatasetInfo(
        id2label=cast(dict[int, str], id2label),
        label2id=cast(dict[str, int], label2id),
        system_prompt=system_prompt,
    )

    return data, info
