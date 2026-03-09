from typing import Literal

type ICFTTask = Literal["seq2seq", "seq-cls", "causal-lm"]
type ICFTDataset = Literal["multinerd"]
type ICFTPrompt = Literal["system", "random", "none"]
type PrefixInit = Literal["pretrained", "labels", "random"]
