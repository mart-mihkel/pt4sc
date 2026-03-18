import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from datasets.splits import Split
from torch.utils.data import Dataset
from transformers import AutoModel, Trainer

from icft.common import init_collate_fn, init_data, init_tokenizer
from icft.datasets.superglue import id2label
from icft.logging import logger


def predict(checkpoint: str):
    path = Path(checkpoint)
    with open(path / "cli_params.json") as f:
        params = json.load(f)

    tokenizer = init_tokenizer(model_path=checkpoint)
    data, _ = init_data(
        tokenizer=tokenizer,
        task=params["task"],
        dataset="superglue",
        prompt_mode=params.get("prompt_mode", "none"),
        workers=params["workers"],
        split=cast(Split, {"test": "test"}),
    )

    test = cast(Dataset, data["test"].remove_columns("labels"))

    logger.debug("load model from checkpoint")
    model = AutoModel.from_pretrained(checkpoint)

    logger.debug("load trainer from checkpoint")
    args = torch.load(path / "training_args.bin", weights_only=False)
    args.eval_strategy = "no"

    collate_fn = init_collate_fn(tokenizer=tokenizer, task=params["task"])
    trainer = Trainer(args=args, model=model, data_collator=collate_fn)

    logger.debug("run predictions")
    res = trainer.predict(test)
    preds = np.argmax(res.predictions, axis=-1)

    out = path / "boolq.jsonl"
    jsonl = [
        json.dumps({"idx": idx, "label": id2label[pred]}) + "\n"
        for idx, pred in zip(data["idx"], preds, strict=True)
    ]

    logger.info("save predictions to %s", out)
    with open(out, "w") as f:
        f.writelines(jsonl)
