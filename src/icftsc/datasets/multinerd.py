from typing import Literal, TypedDict, cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from datasets.utils.info_utils import VerificationMode
from transformers import BatchEncoding, PreTrainedTokenizerFast

from icftsc.constants import bert_model_types, gpt_model_types, t5_model_types
from icftsc.datasets.common import (
    DatasetInfo,
    prepend_system_tokens,
    randomize_prompt,
)
from icftsc.logging import logger
from icftsc.types import Task

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
    ner_tags: list[list[str]]
    lang: list[MultinerdLang]


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
    0: "PER",
    1: "ORG",
    2: "LOC",
    3: "ANIM",
    4: "BIO",
    5: "CEL",
    6: "DIS",
    7: "EVE",
    8: "FOOD",
    9: "INST",
    10: "MEDIA",
    11: "MYTH",
    12: "PLANT",
    13: "TIME",
    14: "VEHI",
}

label2id: dict[MultinerdTag, int] = {
    "PER": 0,
    "ORG": 1,
    "LOC": 2,
    "ANIM": 3,
    "BIO": 4,
    "CEL": 5,
    "DIS": 6,
    "EVE": 7,
    "FOOD": 8,
    "INST": 9,
    "MEDIA": 10,
    "MYTH": 11,
    "PLANT": 12,
    "TIME": 13,
    "VEHI": 14,
}


def _bert_sys_prompt(bos: str, sep: str) -> str:
    return f"{bos}Identify the NER tag of the entity in the sentence.{sep}"


def _bert_prompt(sentence: str, entity: str, bos: str, sep: str, eos: str) -> str:
    return f"{bos}{sentence}{sep}{entity}{eos}"


def _gpt_sys_prompt(bos: str | None) -> str:
    _bos = bos if bos is not None else ""
    return (
        f"{_bos}"
        "You are a Named Entity Recognition (NER) expert. Given a sentence and "
        "a target entity, output the correct entity label. Use exactly one of "
        "the following labels: PER, ORG, LOC, ANIM, BIO, CEL, DIS, EVE, FOOD, "
        "INST, MEDIA, MYTH, PLANT, TIME, VEHI. Respond with only the label and "
        "no explanation.\n\nSentence: Paris is the capital of France.\nEntity: "
        "Paris\nAnswer: LOC\n\n"
    )


def _gpt_prompt(sentence: str, entity: str, bos: str | None) -> str:
    _bos = bos if bos is not None else ""
    return f"{_bos}Sentence: {sentence}\nEntity: {entity}\nAnswer: "


def _t5_sys_prompt() -> str:
    return (
        "Task: NER, identify the NER tag of the entity in the sentence span.\n"
        "Labels: PER ORG, LOC, ANIM, BIO, CEL, DIS, EVE, FOOD, INST, MEDIA, "
        "MYTH, PLANT, TIME, VEHI.\n\nSentence: Paris is the capital of France.\n"
        "Entity: Paris\nAnswer: LOC\n\n"
    )


def _t5_prompt(sentence: str, entity: str) -> str:
    return f"Sentence: {sentence}\nEntity: {entity}\nAnswer: "


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
    batch: MultinerdBatch,
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
) -> BatchEncoding:
    prompts: list[str] = []
    labels: list[int] = []

    for tokens, tag_ids in zip(batch["tokens"], batch["ner_tags"], strict=True):
        sentence = " ".join(tokens)
        tokens, tag_ids = _join_spans(tokens=tokens, tag_ids=tag_ids)

        for token, tag_id in zip(tokens, tag_ids, strict=True):
            if tag_id == -1:
                continue

            prompt = _get_prompt(
                model_type=model_type,
                tokenizer=tokenizer,
                sentence=sentence,
                entity=token,
            )

            prompts.append(prompt)
            labels.append(tag_id)

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
    tag_ids: list[int],
) -> tuple[list[MultinerdTag], list[int]]:
    out_ids = []
    out_tokens = []
    for token, tag_id in zip(tokens, tag_ids, strict=True):
        tag = _id2label_full[tag_id]

        if tag.startswith("B-"):
            tag = cast(MultinerdTag, tag[2:])
            out_ids.append(label2id[tag])
            out_tokens.append(token)
        elif tag.startswith("I-"):
            out_tokens[-1] = f"{out_tokens[-1]} {token}"
        elif tag == "O":
            out_ids.append(-1)
            out_tokens.append(token)
        else:
            tag = cast(MultinerdTag, tag)
            out_ids.append(label2id[tag])
            out_tokens.append(token)

    return out_tokens, out_ids


def _filter_english(batch: MultinerdBatch) -> list[bool]:
    return [lang == "en" for lang in batch["lang"]]


def init_multinerd(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
    workers: int = 0,
    filter_en: bool = True,
    subset: float = 0.1,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    """
    Initialize a modified subset of the MultiNERD dataset.

    The BIO tagging task is converted to a regular NER tagging task by joining
    tokens with B- and I- prefixes into a single span. O tags are dropped
    entirely.

    Each token is split into a separate sample containing the entire context
    sentence and the target token. The task is to classify the tag of the token
    in the entire sequence.
    """
    data = load_dataset(
        "Babelscape/multinerd",
        split=split,
        verification_mode=VerificationMode.NO_CHECKS,
    )

    data = cast(DatasetDict, data)

    if "validation" in data:
        data["dev"] = data.pop("validation")

    if filter_en:
        logger.info("using english only subset")
        data = data.filter(_filter_english, batched=True)

    logger.warning("using %d%% of dev and train data", int(subset * 100))
    if "train" in data:
        idx_train = range(int(subset * len(data["train"])))
        data["train"] = data["train"].select(idx_train)

    if "dev" in data:
        idx_dev = range(int(subset * len(data["dev"])))
        data["dev"] = data["dev"].select(idx_dev)

    data = data.map(
        _tokenize,
        batched=True,
        num_proc=workers,
        remove_columns=["tokens", "ner_tags", "lang"],
        fn_kwargs={"tokenizer": tokenizer, "model_type": model_type, "task": task},
    )

    sys_prompt = _get_sys_prompt(tokenizer=tokenizer, model_type=model_type)
    if "test" in data:
        has_bos = tokenizer.bos_token is not None
        sys = tokenizer(sys_prompt, truncation=True, add_special_tokens=False)
        rand = randomize_prompt(tokenizer=tokenizer, enc=sys)

        logger.info("prepare system propted test sets")
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
        logger.info("%d train samples", len(data["train"]))

    if "dev" in data:
        logger.info("%d dev samples", len(data["dev"]))

    if "test" in data:
        logger.info("%d test samples", len(data["test"]))

    return data, info
