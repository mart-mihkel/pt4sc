import numpy as np
from transformers import EvalPrediction, PreTrainedTokenizerFast

from icft.scripts.common import init_metrics_fn


def test_seq_cls():
    logits = np.array([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    labels = np.array([0, 0])
    eval_pred = EvalPrediction(logits, labels)

    metrics_fn = init_metrics_fn(task="seq-cls")
    metrics = metrics_fn(eval_pred)

    assert metrics["accuracy"] == 1.0


def test_seq2seq(t5_tokenizer: PreTrainedTokenizerFast):
    logits = np.array(
        [
            [[1.0, 4.0, 3.0, 2.0, 5.0], [1.0, 4.0, 3.0, 2.0, 5.0]],
            [[1.0, 4.0, 3.0, 2.0, 5.0], [1.0, 4.0, 3.0, 2.0, 5.0]],
        ]
    )

    labels = np.array([[5, 5], [5, 5]])
    eval_pred = EvalPrediction(logits, labels)

    metrics_fn = init_metrics_fn(task="seq2seq", tokenizer=t5_tokenizer)
    metrics_fn(eval_pred)


def test_causal_lm(gpt2_tokenizer: PreTrainedTokenizerFast):
    logits = np.array(
        [
            [[5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]],
            [[5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]],
        ]
    )

    labels = np.array([[0, 1], [0, 1]])
    eval_pred = EvalPrediction(logits, labels)

    metrics_fn = init_metrics_fn(task="causal-lm", tokenizer=gpt2_tokenizer)
    metrics_fn(eval_pred)
