# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Dict, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from typing import List
from collections import namedtuple


class VoiceGpt(MegatronModule):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):  Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,

    ) -> None:
        super().__init__(config=config)
        self.language_model=GPTModel(        
        config,
        transformer_layer_spec,
        vocab_size,
        max_sequence_length,
        pre_process,
        post_process,
        fp16_lm_cross_entropy,
        parallel_output,
        share_embeddings_and_output_weights,
        position_embedding_type,
        rotary_percent,
        rotary_base)
        # Map (intermediate) vision model outputs to the language model input dimension.

        

        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

        vision_projection_layer_spec = MLPSubmodules (linear_fc1=ColumnParallelLinear)
        

        self.vision_projection = MultimodalProjector(
            config,
            vision_projection_layer_spec,
            'affine',
            # whisper hidden size 1280
            1280,  # input size to the projection.
        )


        self.register_load_state_dict_post_hook(
            _load_state_dict_hook_ignore_param_names
        )
        
    def freeze(
        self, freeze_language_model: bool, freeze_vision_projection: bool
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model:
            modules.append(self.language_model)
        if freeze_vision_projection:
            modules.append(self.vision_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.language_model.set_input_tensor(input_tensor)


    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        whisper_feats: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        audio_embeddings = self.vision_projection(whisper_feats)


        language_embeddings = self.language_model.embedding(
            input_ids=input_ids, position_ids=position_ids
        )
        audio_embeddings=audio_embeddings.permute(1, 0, 2)

        pre_audio_emb = language_embeddings[:11]
        suf_audio_emb = language_embeddings[11+ 1500:]
        conbined_emb = torch.cat([pre_audio_emb, audio_embeddings, suf_audio_emb])
        
        #torch.save(conbined_emb, f'conbined_emb_{torch.distributed.get_rank()}')
        #torch.save(audio_embeddings, f'audio_embeddings_{torch.distributed.get_rank()}')
        #import time
        #time.sleep(5)
        #raise
        # Embedding is computed above so we can discard input and position ids.
        input_ids = None
        position_ids = None
        return self.language_model(
            None,
            None,
            attention_mask,
            conbined_emb,
            labels,
            inference_params,
            packed_seq_params,
            extra_block_kwargs)

    def load_state_dict(self, state_dict, strict=True):
        if True:
            super().load_state_dict(state_dict, strict)
        else:
            self.language_model.load_state_dict(state_dict, strict)

def _load_state_dict_hook_ignore_param_names(
    module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore missing keys during checkpoint loading.

    By default, this should not be used to avoid accidentally missing weights in checkpoint loading.

    Example use case: Use this for the vision projection if you want to load a checkpoint that contains vision and language model weights
    but not the vision projection weights.

    Args:
        param_names (list of str): Parameter names allowed to be missing when calling load_state_dict.
        module (torch.nn.Module): The torch module this hook applies to. Unused here but required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys, which collect the missing and unexpected
            keys when calling load_state_dict on this torch module, respectively.
    """
    #import pdb;pdb.set_trace()
    keys_remove=set()
    for param_name in incompatible_keys.missing_keys:
        if param_name.endswith('_extra_state') or "vision_projection.encoder" in param_name:
            keys_remove.add(param_name)
    for param_name in keys_remove:
        incompatible_keys.missing_keys.remove(param_name)
        #print(param_name)

