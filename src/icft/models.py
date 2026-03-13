import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Parameter
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import (
    CausalLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)

from icft.types import Task


class PTModelConfig(PretrainedConfig):
    model_type = "pt"

    def __init__(
        self,
        task: Task | None = None,
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


class PTEncoderModelConfig(PTModelConfig):
    model_type = "pt_encoder"


class PTEncoderModel(PTModel):
    config_class = PTEncoderModelConfig

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


class PTDecoderModelConfig(PTModelConfig):
    model_type = "pt_decoder"


class PTDecoderModel(PTModel):
    config_class = PTDecoderModelConfig

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> SequenceClassifierOutput | Seq2SeqModelOutput | CausalLMOutput:
        inputs, attn = self._get_prompt(input_ids, attention_mask)
        if self.config.task == "causal-lm":
            labels = self._get_causal_labels(labels)

        out = self.base(
            inputs_embeds=inputs,
            attention_mask=attn,
            labels=labels,
            output_hidden_states=True,
        )

        if self.config.task == "seq-cls":
            return self._post_forward_seq_cls(
                input_ids=input_ids,
                labels=labels,
                last_hidden_state=out.hidden_states[-1],
            )

        return out

    def _post_forward_seq_cls(
        self,
        input_ids: Tensor,
        labels: Tensor,
        last_hidden_state: Tensor,
    ) -> SequenceClassifierOutput:
        prompt_ids = self._get_prompt_ids(input_ids)
        batch_size, seq_len = prompt_ids.shape
        device = input_ids.device

        pad_token_id = self.base.config.pad_token_id
        non_pad_mask = (prompt_ids != pad_token_id).to(device, torch.int32)
        token_indices = torch.arange(seq_len, device=device, dtype=torch.int32)
        last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

        logits = self.base.score(last_hidden_state)
        pooled_logits = logits[
            torch.arange(batch_size, device=device),
            last_non_pad_token,
        ]

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            pooled_logits.view(-1, self.config.num_labels),
            labels.view(-1),
        )

        return SequenceClassifierOutput(loss=loss, logits=pooled_logits)


class PTEncoderDecoderModelConfig(PTModelConfig):
    model_type = "pt_encoder_decoder"


class PTEncoderDecoderModel(PTModel):
    config_class = PTEncoderDecoderModelConfig

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> SequenceClassifierOutput | Seq2SeqModelOutput | CausalLMOutput:
        enc_inputs, enc_attn = self._get_prompt(input_ids, attention_mask)

        dec_input_ids = self._shift_inputs(input_ids)
        dec_attention_mask = self._shift_attention(attention_mask)
        dec_inputs, dec_attn = self._get_prompt(dec_input_ids, dec_attention_mask)

        if self.config.task == "causal-lm":
            labels = self._get_causal_labels(labels)

        if self.config.task == "seq-cls":
            out = self.base.transformer(
                inputs_embeds=enc_inputs,
                attention_mask=enc_attn,
                decoder_inputs_embeds=dec_inputs,
                decoder_attention_mask=dec_attn,
                labels=labels,
            )

            return self._post_forward_seq_cls(
                input_ids=input_ids,
                labels=labels,
                last_hidden_state=out.last_hidden_state,
            )

        return self.base(
            inputs_embeds=enc_inputs,
            attention_mask=enc_attn,
            labels=labels,
        )

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
        lengths = attention_mask.sum(dim=1, keepdim=True).clamp(
            max=attention_mask.size(1) - 1
        )

        return attention_mask.clone().scatter_(1, lengths, 1)

    def _post_forward_seq_cls(
        self,
        input_ids: Tensor,
        labels: Tensor,
        last_hidden_state: Tensor,
    ) -> SequenceClassifierOutput:
        logits = self.base.classification_head(last_hidden_state)
        pooled_logits = logits[:, 0]

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            pooled_logits.view(-1, self.config.num_labels),
            labels.view(-1),
        )

        return SequenceClassifierOutput(loss=loss, logits=pooled_logits)


AutoConfig.register(PTEncoderModelConfig.model_type, PTEncoderModelConfig)
AutoConfig.register(PTDecoderModelConfig.model_type, PTDecoderModelConfig)
AutoConfig.register(PTEncoderDecoderModelConfig.model_type, PTEncoderDecoderModelConfig)

AutoModel.register(PTEncoderModelConfig, PTEncoderModel)
AutoModel.register(PTDecoderModelConfig, PTDecoderModel)
AutoModel.register(PTEncoderDecoderModelConfig, PTEncoderDecoderModel)
