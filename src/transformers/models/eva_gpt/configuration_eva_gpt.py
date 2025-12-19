# coding=utf-8
# Copyright 2025
# Licensed under the Apache License, Version 2.0

"""EvaGPT model configuration"""

from typing import Optional, List

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_rope_utils import RopeParameters


class EvaGptConfig(PreTrainedConfig):
    """
    Configuration class for EvaGPT (GPT-style decoder-only transformer).
    """

    model_type = "eva_gpt"

    # RoPE defaults
    default_theta = 10000.0

    # Pipeline / Tensor parallel hints (optional, advanced)
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        # ---- Core model dims ----
        vocab_size: int = 35303,
        hidden_size: int = 1792,
        intermediate_size: int = 3584,
        num_hidden_layers: int = 18,
        num_attention_heads: int = 14,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,

        # ---- Context / attention ----
        max_position_embeddings: int = 8192,
        sliding_window: int = 4096,
        attention_dropout: float = 0.0,

        # ---- MLP / norms ----
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,

        # ---- MoE / Router ----
        num_local_experts: int = 1,
        num_experts_per_tok: int = 1,
        router_aux_loss_coef: float = 0.01,
        output_router_logits: bool = False,

        # ---- General flags ----
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        layer_types: Optional[List[str]] = None,
        rope_parameters: Optional[RopeParameters] = None,

        # ---- Token ids (LLaMA compatible) ----
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: Optional[int] = None,

        **kwargs,
    ):
        # ---- Core dimensions ----
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )

        # ---- Head dimension consistency ----
        if head_dim is None:
            self.head_dim = hidden_size // num_attention_heads
        else:
            self.head_dim = head_dim

        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim "
                f"({self.num_attention_heads} * {self.head_dim} != {self.hidden_size})"
            )

        # ---- Attention / architecture ----
        self.sliding_window = sliding_window
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range

        # Decoder-only GPT flags
        self.is_decoder = True
        self.add_cross_attention = False
        self.use_cache = use_cache
        self.attention_bias = True

        # ---- MoE / Router ----
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits

        # ---- Layer types (full / sliding attention) ----
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if (i % 2 == 0) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        layer_type_validation(self.layer_types, self.num_hidden_layers)

        # ---- Positional encoding (RoPE / YARN) ----
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = rope_parameters or {
            "rope_type": "yarn",
            "rope_theta": self.default_theta,
            "factor": 2.0,
            "original_max_position_embeddings": 4096,
        }

        # ---- Token ids ----
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )


__all__ = ["EvaGptConfig"]

