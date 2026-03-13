from typing import Literal, TypedDict, cast

from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset
from datasets.splits import Split
from transformers import BatchEncoding, PreTrainedTokenizerFast

from icft.datasets.common import DatasetInfo, init_system_prompt, prepend_system_tokens
from icft.logging import logger
from icft.types import PromptMode, Task

type BoolQALabel = Literal["false", "true"]

id2label: dict[int, BoolQALabel] = {
    0: "false",
    1: "true",
}

label2id: dict[BoolQALabel, int] = {
    "false": 0,
    "true": 1,
}

system_prompt = """Task: Boolean Question Answering

Answer the question based on the given passage. Answer with "true" or "false".

Passage: The Great Wall of China is a series of fortifications made of stone.
Question: Is the Great Wall of China made of stone?
Answer: true

"""


class BoolqBatch(TypedDict):
    passage: list[str]
    question: list[str]
    label: list[int]


def _tokenize_seq_cls(
    batch: BoolqBatch,
    tokenizer: PreTrainedTokenizerFast,
) -> BatchEncoding:
    prompts: list[str] = []
    labels: list[int] = []

    it = zip(batch["passage"], batch["question"], batch["label"])
    for passage, question, label in it:
        prompt = _prompt_template(passage=passage, question=question)
        prompts.append(prompt)
        labels.append(label)

    enc = tokenizer(prompts, add_special_tokens=False, truncation=True)
    enc["labels"] = labels

    return enc


def _tokenize_seq2seq(
    batch: BoolqBatch,
    tokenizer: PreTrainedTokenizerFast,
) -> BatchEncoding:
    prompts: list[str] = []
    labels: list[str] = []

    it = zip(batch["passage"], batch["question"], batch["label"])
    for passage, question, label in it:
        prompt = _prompt_template(passage=passage, question=question)
        prompts.append(prompt)
        label_str = "true" if label == 1 else "false"
        labels.append(f"{label_str}{tokenizer.eos_token}")

    enc = tokenizer(prompts, add_special_tokens=False, truncation=True)
    labels_enc = tokenizer(labels, add_special_tokens=False, truncation=True)
    enc["labels"] = labels_enc["input_ids"]

    return enc


def _tokenize_causal_lm(
    batch: BoolqBatch,
    tokenizer: PreTrainedTokenizerFast,
) -> BatchEncoding:
    ids: list[list[int]] = []
    attn: list[list[int]] = []
    labels: list[list[int]] = []

    it = zip(batch["passage"], batch["question"], batch["label"])
    for passage, question, label in it:
        prompt = _prompt_template(passage=passage, question=question)
        label_str = "true" if label == 1 else "false"
        answer = f"{label_str}{tokenizer.eos_token}"

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


def _prompt_template(passage: str, question: str) -> str:
    return f"Passage: {passage}\nQuestion: {question}\nAnswer: "


def init_superglue(
    tokenizer: PreTrainedTokenizerFast,
    task: Task,
    prompt_mode: PromptMode,
    workers: int,
    subset: str = "boolq",
    split: Split | None = None,
) -> tuple[DatasetDict, DatasetInfo]:
    data = cast(DatasetDict, load_dataset("aps/super_glue", subset, split=split))

    if "validation" in data:
        data["dev"] = data.pop("validation")

    logger.debug("tokenize superglue (%s) for %s", subset, task)
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
            fn_kwargs=dict(sys=sys),
            num_proc=workers,
        )

    info = DatasetInfo(
        id2label=cast(dict[int, str], id2label),
        label2id=cast(dict[str, int], label2id),
        system_prompt=system_prompt,
    )

    return data, info
