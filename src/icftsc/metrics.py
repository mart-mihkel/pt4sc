import evaluate
import numpy as np
from scipy.special import log_softmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.fft import Tensor
from transformers import EvalPrediction, PreTrainedTokenizerFast

from icftsc.logging import logger

_bleu = evaluate.load("bleu")
_rouge = evaluate.load("rouge")

# global state during `trainer.evaluate` to collect batched eval loop outputs
_labels: list[np.ndarray] = []
_preds: list[np.ndarray] = []


def _update_state(eval_pred: EvalPrediction):
    global _labels, _preds

    batch_labels = eval_pred.label_ids
    if isinstance(batch_labels, tuple):
        batch_labels = batch_labels[0]

    if isinstance(batch_labels, Tensor):
        batch_labels = batch_labels.detach().cpu().numpy()

    batch_logits = eval_pred.predictions
    if isinstance(batch_logits, tuple):
        batch_logits = batch_logits[0]

    if isinstance(batch_logits, Tensor):
        batch_logits = batch_logits.detach().cpu().numpy()

    batch_preds = np.argmax(batch_logits, axis=-1)

    _labels.append(batch_labels)
    _preds.append(batch_preds)


def _collect_state() -> tuple[np.ndarray, np.ndarray]:
    global _labels, _preds

    return np.concat(_labels, axis=0), np.concat(_preds, axis=0)


def _reset_state():
    global _labels, _preds

    _labels = []
    _preds = []


def _compute_classification_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
) -> dict[str, float]:
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
        logger.warning("BLEU evaluation was run in a child process")
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
        logger.warning("ROUGE evaluation was run in a child process")
        return {}

    return res


def compute_metrics_seq_cls(
    eval_pred: EvalPrediction,
    compute_result: bool = True,
) -> dict[str, float]:
    _update_state(eval_pred)

    if not compute_result:
        return {}

    labels, preds = _collect_state()
    mask = labels != -100

    metrics = _compute_classification_metrics(labels[mask], preds[mask])

    _reset_state()

    return metrics


def compute_metrics_seq2seq(
    eval_pred: EvalPrediction,
    tokenizer: PreTrainedTokenizerFast,
    compute_result: bool = True,
) -> dict[str, float]:
    _update_state(eval_pred)

    if not compute_result:
        return {}

    labels, preds = _collect_state()
    mask = labels != -100

    simple = _compute_classification_metrics(labels[mask], preds[mask])
    bleu = _compute_bleu(labels[mask], preds[mask], tokenizer)
    rouge = _compute_rouge(labels[mask], preds[mask], tokenizer)

    _reset_state()

    return simple | bleu | rouge


def compute_metrics_causal_lm(
    eval_pred: EvalPrediction,
    tokenizer: PreTrainedTokenizerFast,
    compute_result: bool = True,
) -> dict[str, float]:
    _update_state(eval_pred)

    if not compute_result:
        return {}

    # TODO: accumulate logits for perplexity
    labels, preds = _collect_state()
    mask = labels != -100

    simple = _compute_classification_metrics(labels[mask], preds[mask])
    bleu = _compute_bleu(labels[mask], preds[mask], tokenizer)
    rouge = _compute_rouge(labels[mask], preds[mask], tokenizer)

    _reset_state()

    return simple | bleu | rouge
