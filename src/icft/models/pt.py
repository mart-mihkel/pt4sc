import torch
from torch import Tensor
from torch.nn import Parameter
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
    AutoModelForCausalLM,
)
from transformers.modeling_outputs import (
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
    CausalLMOutput,
)

from icft.types import ICFTTask


class PTModelConfig(PretrainedConfig):
    model_type = "pt"

    def __init__(
        self,
        task: ICFTTask | None = None,
        num_virtual_tokens: int = 0,
        pretrained_model: str | None = None,
        num_labels: int = 1,
        id2label: dict[int, str] = {0: "LABEL_0"},
        label2id: dict[str, int] = {"LABEL_0": 0},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task = task
        self.num_virtual_tokens = num_virtual_tokens
        self.pretrained_model = pretrained_model
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id


class PTModel(PreTrainedModel):
    config_class = PTModelConfig

    def __init__(self, config: PTModelConfig) -> None:
        super().__init__(config)

        self.config = config
        _config = AutoConfig.from_pretrained(
            config.pretrained_model,
            num_labels=config.num_labels,
            id2label=config.id2label,
            label2id=config.label2id,
        )

        if config.task is None:
            raise ValueError("No task specified")
        elif config.task == "seq2seq":
            base = AutoModelForSeq2SeqLM.from_config(_config)
        elif config.task == "seq-cls":
            base = AutoModelForSequenceClassification.from_config(_config)
        elif config.task == "causal-lm":
            base = AutoModelForCausalLM.from_config(_config)
        else:
            raise NotImplementedError(f"Task '{config.task}'")

        emb_dim = base.get_input_embeddings().embedding_dim
        prefix = torch.randn(config.num_virtual_tokens, emb_dim)

        self.base = base
        self.prefix = Parameter(prefix)

        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> SequenceClassifierOutput | Seq2SeqModelOutput | CausalLMOutput:
        inputs, attn = self._get_prompt(input_ids, attention_mask)
        if self.config.task == "causal-lm":
            labels = self._get_causal_labels(labels)

        return self.base(inputs_embeds=inputs, attention_mask=attn, labels=labels)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base, "gradient_checkpointing_enable"):
            self.base.gradient_checkpointing_enable(**kwargs)
        else:
            raise ValueError("Base model does not support gradient checkpointing")

    def gradient_checkpointing_disable(self):
        if hasattr(self.base, "gradient_checkpointing_disable"):
            self.base.gradient_checkpointing_disable()
        else:
            raise ValueError("Base model does not support gradient checkpointing")

    def _get_prompt(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        num_virtual = self.config.num_virtual_tokens
        batch_size = input_ids.shape[0]
        device = input_ids.device

        prompt_emb = self.base.get_input_embeddings()(input_ids)
        prefix_emb = self.prefix.expand(batch_size, -1, -1)
        prefix_attn = torch.ones(
            batch_size,
            num_virtual,
            device=device,
            dtype=attention_mask.dtype,
        )

        inputs_embeds = torch.cat([prefix_emb, prompt_emb], dim=1)
        attention_mask = torch.cat([prefix_attn, attention_mask], dim=1)

        return inputs_embeds, attention_mask

    def _get_prompt_ids(self, input_ids: Tensor) -> Tensor:
        batch_size = input_ids.shape[0]

        prefix_ids = torch.full(
            (batch_size, self.config.num_virtual_tokens),
            100,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

        return torch.cat([prefix_ids, input_ids], dim=1)

    def _get_causal_labels(self, labels: Tensor) -> Tensor:
        prefix_labels = torch.full(
            (labels.shape[0], self.config.num_virtual_tokens),
            -100,
            device=labels.device,
            dtype=labels.dtype,
        )

        return torch.cat([prefix_labels, labels], dim=1)
