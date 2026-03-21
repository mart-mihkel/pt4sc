from typing import Literal, TypedDict, cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from transformers import BatchEncoding, PreTrainedTokenizerFast

from icftsc.constants import bert_model_types, gpt_model_types, t5_model_types
from icftsc.datasets.common import (
    DatasetInfo,
    prepend_system_tokens,
    randomize_prompt,
)
from icftsc.logging import logger
from icftsc.types import Task

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


class EstnerBatch(TypedDict):
    doc_id: list[int]
    sent_id: list[int]
    tokens: list[list[str]]
    ner_tags: list[list[str]]
    ner_tags1: list[list[str]]
    ner_tags2: list[list[str]]


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


def _bert_sys_prompt(bos: str, sep: str) -> str:
    return f"{bos}Määra nimeüksuse NER märgen lauses.{sep}"


def _bert_prompt(sentence: str, entity: str, bos: str, sep: str, eos: str) -> str:
    return f"{bos}{sentence}{sep}{entity}{eos}"


def _gpt_sys_prompt(bos: str | None) -> str:
    _bos = bos if bos is not None else ""
    return (
        f"{_bos}"
        "Sa oled nimeüksuste tuvastamise (NER) ekspert. Sulle antakse lause ja "
        "sihtüksus ning pead tagastama õige märgendi. Kasuta täpselt ühte "
        "järgmistest märgenditest: PER, ORG, LOC, GPE, PROD, EVENT, DATE, TIME "
        "TITLE, MONEY, PERCENT, O. Vasta ainult märgendiga ilma selgituseta.\n\n"
        "Lause: Pariis on Prantusmaa pealinn.\nSihtüksus: Pariis\nMärgend: "
        "LOC\n\n"
    )


def _gpt_prompt(sentence: str, entity: str, bos: str | None) -> str:
    _bos = bos if bos is not None else ""
    return f"{_bos}Lause: {sentence}\nÜksus: {entity}\nVastus: "


def _t5_sys_prompt() -> str:
    return (
        "Ülesanne: NER, tuvasta lauses oleva sihtüksuse NER-märgend.\nMärgendid: "
        "PER, ORG, LOC, GPE, PROD, EVENT, DATE, TIME, TITLE, MONEY, PERCENT, "
        "O.\n\nLause: Pariis on Prantusmaa pealinn.\nSihtüksus: Pariis\nMärgend: "
        "LOC\n\n"
    )


def _t5_prompt(sentence: str, entity: str) -> str:
    return f"Lause: {sentence}\nÜksus: {entity}\nVastus: "


def _get_sys_prompt(tokenizer: PreTrainedTokenizerFast, model_type: str) -> str:
    if model_type in bert_model_types:
        return _bert_sys_prompt(bos=tokenizer.bos_token, sep=tokenizer.sep_token)

    if model_type in gpt_model_types:
        return _gpt_sys_prompt(bos=tokenizer.bos_token)

    if model_type in t5_model_types:
        return _t5_sys_prompt()

    raise NotImplementedError(f"Model type '{model_type}'")


def _get_prompt(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    sentence: str,
    entity: str,
) -> str:
    if model_type in bert_model_types:
        return _bert_prompt(
            sentence=sentence,
            entity=entity,
            bos=tokenizer.bos_token,
            sep=tokenizer.sep_token,
            eos=tokenizer.eos_token,
        )

    if model_type in gpt_model_types:
        return _gpt_prompt(sentence=sentence, entity=entity, bos=tokenizer.bos_token)

    if model_type in t5_model_types:
        return _t5_prompt(sentence=sentence, entity=entity)

    raise NotImplementedError(f"Model type '{model_type}'")


def _tokenize(
    batch: EstnerBatch,
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
) -> BatchEncoding:
    prompts: list[str] = []
    labels: list[int] = []

    for tokens, tags in zip(batch["tokens"], batch["ner_tags"], strict=True):
        sentence = " ".join(tokens)
        tokens, tags = _join_spans(tokens=tokens, tags=tags)

        for token, tag in zip(tokens, tags, strict=True):
            prompt = _get_prompt(
                model_type=model_type,
                tokenizer=tokenizer,
                sentence=sentence,
                entity=token,
            )

            prompts.append(prompt)
            labels.append(label2id[tag])

    enc = tokenizer(prompts, truncation=True, add_special_tokens=False)
    if task == "seqcls":
        enc["labels"] = labels
    elif task == "causal":
        enc["labels"] = [
            [-100] * len(prompt_ids)
            + tokenizer.encode(id2label[tag_id])
            + [tokenizer.eos_token_id]
            for prompt_ids, tag_id in zip(enc["input_ids"], labels, strict=True)
        ]
    elif task == "seq2seq":
        enc["labels"] = [
            [*tokenizer.encode(id2label[tag_id]), tokenizer.eos_token_id]
            for tag_id in labels
        ]
    else:
        raise NotImplementedError(f"Task '{task}'")

    return enc


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


def init_estner(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
    workers: int = 0,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    """
    Initialize a modified version of the EstNER dataset.

    The BIO tagging task is converted to a regular NER tagging task by joining
    tokens with B- and I- prefixes into a single span.

    Each token is split into a separate sample containing the entire context
    sentence and the target token. The task is to classify the tag of the token
    in the entire sequence.
    """
    data = cast(DatasetDict, load_dataset("tartuNLP/EstNER", split=split))

    cols = ["doc_id", "sent_id", "tokens", "ner_tags", "ner_tags_2", "ner_tags_3"]
    data = data.map(
        _tokenize,
        batched=True,
        num_proc=workers,
        remove_columns=cols,
        fn_kwargs={"tokenizer": tokenizer, "model_type": model_type, "task": task},
    )

    sys_prompt = _get_sys_prompt(tokenizer=tokenizer, model_type=model_type)
    if "test" in data:
        has_bos = tokenizer.bos_token is not None
        sys = tokenizer(sys_prompt, truncation=True, add_special_tokens=False)
        rand = randomize_prompt(tokenizer=tokenizer, enc=sys)

        logger.debug("prepare system propted test sets")
        data["test-system"] = data["test"].map(
            prepend_system_tokens,
            batched=True,
            num_proc=workers,
            fn_kwargs={"sys": sys, "has_bos": has_bos},
        )

        data["test-random"] = data["test"].map(
            prepend_system_tokens,
            batched=True,
            num_proc=workers,
            fn_kwargs={"sys": rand, "has_bos": has_bos},
        )

    info = DatasetInfo(
        id2label=cast(dict[int, str], id2label),
        label2id=cast(dict[str, int], label2id),
        system_prompt=sys_prompt,
    )

    if "train" in data:
        logger.debug("%d train samples", len(data["train"]))

    if "dev" in data:
        logger.debug("%d dev samples", len(data["dev"]))

    if "test" in data:
        logger.debug("%d test samples", len(data["test"]))

    return data, info
