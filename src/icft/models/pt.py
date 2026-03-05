from typing import cast
import torch
from torch import Tensor, FloatTensor
from torch.nn import Parameter, CrossEntropyLoss
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
            raise NotImplementedError(f"task {config.task} is not implemented")

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
        num_virtual = self.config.num_virtual_tokens
        batch_size = input_ids.shape[0]
        device = input_ids.device

        emb = self.base.get_input_embeddings()

        prompt_emb = emb(input_ids)
        prefix_emb = self.prefix.expand(batch_size, -1, -1)
        prefix_attn = torch.ones(
            batch_size,
            num_virtual,
            device=device,
            dtype=attention_mask.dtype,
        )

        inputs = torch.cat([prefix_emb, prompt_emb], dim=1)
        attn = torch.cat([prefix_attn, attention_mask], dim=1)

        if self.config.task == "causal-lm":
            prefix_labels = torch.full(
                (batch_size, num_virtual),
                -100,
                device=device,
                dtype=labels.dtype,
            )

            labels = torch.cat([prefix_labels, labels], dim=1)

        out = self.base(
            inputs_embeds=inputs,
            attention_mask=attn,
            labels=labels,
            output_hidden_states=True,
        )

        if self.config.task == "seq-cls" and self.base.config.model_type == "gpt2":
            prefix_ids = torch.zeros(
                batch_size,
                num_virtual,
                device=device,
                dtype=input_ids.dtype,
            )

            input_ids = torch.cat([prefix_ids, input_ids], dim=1)
            return self._forward_gpt2_seq_cls(
                input_ids=input_ids,
                labels=labels,
                out=out,
            )
        else:
            return out

    def _forward_gpt2_seq_cls(
        self,
        input_ids: Tensor,
        labels: Tensor,
        out: SequenceClassifierOutput,
    ) -> SequenceClassifierOutput:
        pad_token_id = self.base.config.pad_token_id
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[-1]
        device = input_ids.device

        hidden_states = out.hidden_states
        if hidden_states is None:
            raise ValueError("Base model didn't retrun hidden states")

        non_pad_mask = (input_ids != pad_token_id).to(device, torch.int32)
        token_indices = torch.arange(seq_len, device=device, dtype=torch.int32)
        last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

        logits = self.base.score(hidden_states[-1])
        pooled_logits = logits[
            torch.arange(batch_size, device=device),
            last_non_pad_token,
        ]

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(pooled_logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=cast(FloatTensor, pooled_logits),
        )

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base, "gradient_checkpointing_enable"):
            self.base.gradient_checkpointing_enable(**kwargs)
        else:
            raise ValueError("Base model does not support gradient checkpointing")

    def gradient_checkpointing_disable(self):
        if hasattr(self.base, "gradient_checkpointing_disable"):
            self.base.gradient_checkpointing_disable()
