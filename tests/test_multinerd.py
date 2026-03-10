from transformers import PreTrainedTokenizerFast

from icft.datasets.multinerd import init_multinerd

split = {
    "train": "train[:100]",
    "validation": "validation[:100]",
    "test": "test[:100]",
}


def test_multinerd_mmbert(mmbert_tokenizer: PreTrainedTokenizerFast):
    data, _ = init_multinerd(
        tokenizer=mmbert_tokenizer,
        task="seq-cls",
        prompt_mode="system",
        filter_en=False,
        workers=0,
        split=split,  # type: ignore
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0


def test_multinerd_gpt2(gpt2_tokenizer):
    data, _ = init_multinerd(
        tokenizer=gpt2_tokenizer,
        task="causal-lm",
        prompt_mode="system",
        filter_en=False,
        workers=0,
        split=split,  # type: ignore
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0


def test_multinerd_t5(t5_tokenizer):
    data, _ = init_multinerd(
        tokenizer=t5_tokenizer,
        task="seq2seq",
        prompt_mode="system",
        filter_en=False,
        workers=0,
        split=split,  # type: ignore
    )

    assert len(data["train"]) > 0
    assert len(data["dev"]) > 0
    assert len(data["test"]) > 0
