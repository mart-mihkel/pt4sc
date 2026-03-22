import torch
from torch import Tensor
from torch.nn import Parameter
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.utils.generic import ModelOutput

from icftsc.logging import logger
from icftsc.types import Task


class PTModelConfig(PretrainedConfig):
    model_type = "pt"

    def __init__(
        self,
        num_virtual_tokens: int = 0,
        pretrained_model: str | None = None,
        task: Task | None = None,
        num_labels: int = 1,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_virtual_tokens = num_virtual_tokens
        self.pretrained_model = pretrained_model
        self.task = task
        self.num_labels = num_labels
        self.id2label = id2label if id2label is not None else {0: "LABEL_0"}
        self.label2id = label2id if label2id is not None else {"LABEL_0": 0}


class PTModel(PreTrainedModel):
    config_class = PTModelConfig

    def __init__(self, config: PTModelConfig) -> None:
        super().__init__(config)

        base_config = AutoConfig.from_pretrained(
            config.pretrained_model,
            num_labels=config.num_labels,
            id2label=config.id2label,
            label2id=config.label2id,
        )

        if config.task == "seqcls":
            logger.debug("init empty pretrained model for sequence classification")
            base = AutoModelForSequenceClassification.from_config(base_config)
        elif config.task == "causal":
            logger.debug("init empty pretrained model for causal language modeling")
            base = AutoModelForCausalLM.from_config(base_config)
        elif config.task == "seq2seq":
            logger.debug("init empty pretrained model for sequence to sequence")
            base = AutoModelForSeq2SeqLM.from_config(base_config)
        else:
            raise NotImplementedError(f"Task '{config.task}'")

        emb_dim = base.get_input_embeddings().embedding_dim
        prefix = torch.randn(config.num_virtual_tokens, emb_dim)

        if "n_positions" in base_config:
            self.max_pos = base_config.n_positions
        elif "max_position_embeddings" in base_config:
            self.max_pos = base_config.max_position_embeddings
        else:
            logger.warning("not detecting maximum input sequence length")
            self.max_pos = float("inf")

        self.config = config
        self.base = base
        self.prefix = Parameter(prefix)
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> ModelOutput:
        raise NotImplementedError("Forward pass of abstract model")

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

        max_input_len = self.max_pos - num_virtual
        if input_ids.shape[1] > max_input_len:
            logger.warning("prefixed input sequence length exceeds model maximum")
            input_ids = input_ids[:, :max_input_len]
            attention_mask = attention_mask[:, :max_input_len]

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
            0,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

        return torch.cat([prefix_ids, input_ids], dim=1)

    def _get_causal_labels(self, labels: Tensor) -> Tensor:
        batch_size = labels.shape[0]
        prefix_labels = torch.full(
            (batch_size, self.config.num_virtual_tokens),
            -100,
            device=labels.device,
            dtype=labels.dtype,
        )

        return torch.cat([prefix_labels, labels], dim=1)

    def _shift_inputs(self, input_ids: Tensor) -> Tensor:
        decoder_start_token_id = self.base.config.decoder_start_token_id
        pad_token_id = self.base.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError("No decoder start token id")

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("No pad token id")

        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def _shift_attention(self, attention_mask: Tensor) -> Tensor:
        maxlen = attention_mask.size(1) - 1
        lengths = attention_mask.sum(dim=1, keepdim=True).clamp(max=maxlen)
        return attention_mask.clone().scatter_(1, lengths, 1)


AutoConfig.register(PTModelConfig.model_type, PTModelConfig)
