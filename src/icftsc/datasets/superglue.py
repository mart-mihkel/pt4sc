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

type BoolQALabel = Literal["false", "true"]


class BoolqBatch(TypedDict):
    idx: list[int]
    passage: list[str]
    question: list[str]
    label: list[int]


id2label: dict[int, BoolQALabel] = {
    0: "false",
    1: "true",
}

label2id: dict[BoolQALabel, int] = {
    "false": 0,
    "true": 1,
}


def _bert_sys_prompt(bos: str, sep: str) -> str:
    return f"{bos}Identify the NER tag of the entity in the sentence.{sep}"


def _bert_prompt(question: str, passage: str, bos: str, sep: str, eos: str) -> str:
    return f"{bos}{question}{sep}{passage}{eos}"


def _gpt_sys_prompt(bos: str | None) -> str:
    _bos = bos if bos is not None else ""
    return (
        f"{_bos}"
        "You are a Boolean Question Answering expert.Given a passage and a "
        'question, answer with exactly one word: "true" or "false".\nDo '
        "not provide any explanation.\n\nPassage: The Great Wall of China is a "
        "series of fortifications made of stone.\nQuestion: Is the Great Wall "
        "of China made of stone?\nAnswer: true\n\n"
    )


def _gpt_prompt(question: str, passage: str, bos: str | None) -> str:
    _bos = bos if bos is not None else ""
    return f"{_bos}passage: {passage}\nquestion: {question}\nanswer: "


def _t5_sys_prompt() -> str:
    return (
        "boolean question answering: given a passage and a question, "
        'answer with "true" or "false".\npassage: The Great Wall of '
        "China is a series of fortifications made of stone.\nquestion: Is the "
        "Great Wall of China made of stone?\nanswer: true\n"
    )


def _t5_prompt(question: str, passage: str) -> str:
    return f"Passage: {passage}\nQuestion: {question}\nAnswer: "


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
    question: str,
    passage: str,
) -> str:
    if model_type in bert_model_types:
        return _bert_prompt(
            question=question,
            passage=passage,
            bos=tokenizer.bos_token,
            sep=tokenizer.sep_token,
            eos=tokenizer.eos_token,
        )

    if model_type in gpt_model_types:
        return _gpt_prompt(question=question, passage=passage, bos=tokenizer.bos_token)

    if model_type in t5_model_types:
        return _t5_prompt(question=question, passage=passage)

    raise NotImplementedError(f"Model type '{model_type}'")


def _tokenize(
    batch: BoolqBatch,
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
) -> BatchEncoding:
    prompts: list[str] = []
    labels: list[int] = []

    it = zip(batch["passage"], batch["question"], batch["label"], strict=True)
    for passage, question, label_id in it:
        prompt = _get_prompt(
            model_type=model_type,
            tokenizer=tokenizer,
            question=question,
            passage=passage,
        )

        prompts.append(prompt)
        labels.append(label_id)

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


def init_superglue(
    tokenizer: PreTrainedTokenizerFast,
    model_type: str,
    task: Task,
    workers: int = 0,
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    data = cast(DatasetDict, load_dataset("aps/super_glue", "boolq", split=split))

    if "validation" in data:
        data["dev"] = data.pop("validation")

    data = data.map(
        _tokenize,
        batched=True,
        num_proc=workers,
        remove_columns=["question", "passage", "label"],
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
        logger.info("%d train samples", len(data["train"]))

    if "dev" in data:
        logger.info("%d dev samples", len(data["dev"]))

    if "test" in data:
        logger.info("%d test samples", len(data["test"]))

    return data, info
