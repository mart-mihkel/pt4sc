import evaluate
import numpy as np
from scipy.special import log_softmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import EvalPrediction, PreTrainedTokenizerFast

from icft.logging import logger

_bleu = evaluate.load("bleu")
_rouge = evaluate.load("rouge")


def _compute_simple_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _compute_perplexity(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    idx = np.arange(labels.shape[0])
    log_probs = log_softmax(logits, axis=-1)[idx, labels]
    perplexity = np.exp(-log_probs.mean())
    return {"perplexity": perplexity}


def _compute_bleu(
    labels: np.ndarray,
    preds: np.ndarray,
    tokenizer: PreTrainedTokenizerFast,
) -> dict[str, float]:
    predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)
    res = _bleu.compute(predictions=predictions, references=references)  # type: ignore
    if res is None:
        logger.warning("BLEU evaluation was no run on the main process")
        return {}

    return {"bleu": res["bleu"]}


def _compute_rouge(
    labels: np.ndarray,
    preds: np.ndarray,
    tokenizer: PreTrainedTokenizerFast,
) -> dict[str, float]:
    predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)
    res = _rouge.compute(predictions=predictions, references=references)  # type: ignore
    if res is None:
        logger.warning("ROUGE evaluation was no run on the main process")
        return {}

    return res


def compute_metrics_seq_cls(eval_pred: EvalPrediction) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return _compute_simple_metrics(labels, preds)


def compute_metrics_seq2seq(
    eval_pred: EvalPrediction,
    tokenizer: PreTrainedTokenizerFast,
) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100

    simple = _compute_simple_metrics(labels[mask], preds[mask])
    bleu = _compute_bleu(labels[mask], preds[mask], tokenizer)
    rouge = _compute_rouge(labels[mask], preds[mask], tokenizer)

    return simple | bleu | rouge


def compute_metrics_causal_lm(
    eval_pred: EvalPrediction,
    tokenizer: PreTrainedTokenizerFast,
) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100

    simple = _compute_simple_metrics(labels[mask], preds[mask])
    perplexity = _compute_perplexity(labels, logits)
    bleu = _compute_bleu(labels[mask], preds[mask], tokenizer)
    rouge = _compute_rouge(labels[mask], preds[mask], tokenizer)

    return simple | perplexity | bleu | rouge
