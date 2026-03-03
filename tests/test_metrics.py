from typing import cast

import numpy as np
from transformers import AutoTokenizer, EvalPrediction, PreTrainedTokenizerFast

from icft.common import init_metrics_fn


def test_seq_cls():
    logits = np.array([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    labels = np.array([0, 0])
    eval_pred = cast(EvalPrediction, (logits, labels))

    metrics_fn = init_metrics_fn(task="seq-cls")
    metrics = metrics_fn(eval_pred)

    assert metrics["accuracy"] == 1.0


def test_seq2seq():
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("google-t5/t5-small"),
    )

    logits = np.array(
        [
            [[1.0, 4.0, 3.0, 2.0, 5.0], [1.0, 4.0, 3.0, 2.0, 5.0]],
            [[1.0, 4.0, 3.0, 2.0, 5.0], [1.0, 4.0, 3.0, 2.0, 5.0]],
        ]
    )

    labels = np.array([[5, 5], [5, 5]])
    eval_pred = cast(EvalPrediction, (logits, labels))

    metrics_fn = init_metrics_fn(task="seq2seq", tokenizer=tokenizer)
    metrics_fn(eval_pred)


def test_causal_lm():
    tokenizer = cast(
        PreTrainedTokenizerFast,
        AutoTokenizer.from_pretrained("openai-community/gpt2"),
    )

    logits = np.array(
        [
            [[5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]],
            [[5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]],
        ]
    )

    labels = np.array([[0, 1], [0, 1]])
    eval_pred = cast(EvalPrediction, (logits, labels))

    metrics_fn = init_metrics_fn(task="causal-lm", tokenizer=tokenizer)
    metrics_fn(eval_pred)
