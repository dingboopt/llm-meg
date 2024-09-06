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
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from typing import List
from collections import namedtuple
from transformers import WhisperForConditionalGeneration


IMAGE_TOKEN_INDEX = -200  # ID for images in the input sequence.
IGNORE_INDEX = -100  # ID for labels that should be ignored.

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
        **kwargs

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

        
        # one adapter for now
        adapter = kwargs['adapter_list'][0]
        adapter_type, input_size = adapter.split(':')
        input_size = int(input_size)

            
        

        vision_projection_layer_spec = MLPSubmodules(
            linear_fc1=ColumnParallelLinear,
            linear_fc2=RowParallelLinear,
        )
        

        self.vision_projection = MultimodalProjector(
            config,
            vision_projection_layer_spec,
            adapter_type,
            # whisper hidden size 1280
            input_size,  # input size to the projection.
        )


        self.register_load_state_dict_post_hook(
            _load_state_dict_hook_ignore_param_names
        )
        
        #encoder_path = '/workspace/sft/yc-speech/models_hub/models/ssl/whisper/large-v3'
        encoder_path=kwargs['encoder_path']
        current_device_index = torch.cuda.current_device()
        # 获取当前设备的对象
        current_device = torch.device(f"cuda:{current_device_index}")
        self.whisper_encoder = WhisperForConditionalGeneration.from_pretrained(
            encoder_path, device_map=current_device, torch_dtype=torch.bfloat16
        ).get_encoder()
        
        ########### for now, only tp enabled
        self.add_decoder = True
        self.pre_process = True
        ##########3
        self._img_seq_len = 1500
        
    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.language_model.set_input_tensor(input_tensor)
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



    def _preprocess_data(
        self,
        image_embeddings,
        language_embeddings,
        input_ids,
        loss_mask,
        labels,
        use_inference_kv_cache,
        image_token_index,
        num_image_tiles,
    ):
        """Preprocess input data before input to language model.

        This function is adopted from
        https://github.com/huggingface/transformers/blob/85817d98fb60977c97e3014196a462b732d2ed1a/src/transformers/models/llava_next/modeling_llava_next.py#L409
        for our input data conventions.

        image_token_index = -200 indicates the image position in the input_ids = [0, 1, -200, 2, 3] and labels = [1, -200, 2, 3, 4], for example.
        We want to replace the image position (-200) with image_embeddings and return the following:
        - final_embeddings = [0, 1, image_embeddings, 2, 3],
        - final_labels = [1, -100, 2, 3, 4]
        - final_loss_mask = [1, 0, 0, 1, 1]

        This function also handles the case where the input does not contain an image (text-only sample). It also handles the case where a single input
        image is split into multiple tiles.

        If pipeline parallelism is not used, then self.pre_process and self.post_process are both True and we update both
        input embeddings, labels and loss masks (if available).

        If pipeline parallelism is used, then we do the following
        - the first language model chunk has self.pre_process = True and self.post_process = False. We update input embeddings.
        - the middle language model chunk(s) has self.pre_process = False and self.post_process = False. We don't need to update anything.
        - the last language model chunk has self.pre_process = False and self.post_process = True. We update labels and loss mask.

        TODO: This function should adjust the attention mask too. Currently, we assume the language model uses a causal mask.

        Returns:
            final_embedding (torch.Tensor): image and text embeddings concated [combined_seq_len, b, h].
            final_labels (torch.Tensor): labels for image and text positions [b, combined_seq_len].
            final_loss_mask (torch.Tensor): loss mask for image and text positions [b, combined_seq_len].
        """
        assert self.add_decoder, "input text preprocessing is only needed for the language model"

        # No pre- or postprocessing needed. With pipeline parallel > 2, this means a chunk in the middle of the model.
        if not self.pre_process and not self.post_process:
            return language_embeddings, loss_mask, labels

        # If using the inference KV cache, the image tokens are already computed.
        if use_inference_kv_cache:
            return language_embeddings, loss_mask, labels

        img_seq_len = self._img_seq_len
        batch_size, text_seq_len = input_ids.shape

        has_labels = labels is not None
        if has_labels:
            assert (
                labels.shape == loss_mask.shape
            ), f"mismatching labels shape {labels.shape} and loss mask shape {loss_mask.shape}"

        # Create indices for new text and label positions.
        with torch.no_grad():
            image_token_mask = input_ids == image_token_index
            num_images_per_sample = torch.sum(image_token_mask, dim=-1)

            # Number of tiles per sample.
            num_image_tiles_batch = num_image_tiles.split(num_images_per_sample.tolist(), dim=0)
            num_image_tiles_batch = torch.tensor(
                [x.sum() for x in num_image_tiles_batch], device=input_ids.device
            )

            # Sequence length for each sample is the image sequence length multiplied by the number of tiles for that image, minus image token indices,
            # plus text sequence length.
            seq_lens = num_image_tiles_batch * img_seq_len - num_images_per_sample + text_seq_len
            max_seq_len = seq_lens.max()
            batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

            # New position ids for the text tokens, shifted by the image sequence length.
            # E.g. for input_ids = [-200, 1, 2, 3] and img_seq_len = 576, we get new_position_ids = [576, 577, 578, 579].
            # text_position_ids are then [577, 578, 579].
            image_token_mask_lens = image_token_mask.int().clone()
            # -1 is for the removed image token index.
            image_token_mask_lens[image_token_mask] = num_image_tiles * img_seq_len - 1
            # +1 is needed here for the cumulative sum. -1 is adjusting for zero-based indexing.
            new_position_ids = torch.cumsum((image_token_mask_lens + 1), dim=-1) - 1
            text_position_ids = new_position_ids[batch_indices, non_image_indices]

            # Labels are shifted to left by one. So, shift text position ids and non-image indices to left by one.
            if has_labels:
                label_text_position_ids = text_position_ids - 1
                valid_label_text_position_ids = label_text_position_ids >= 0
                label_text_position_ids = label_text_position_ids[valid_label_text_position_ids]

                label_batch_indices = batch_indices[valid_label_text_position_ids]

                label_non_image_indices = non_image_indices - 1
                valid_label_non_image_indices = label_non_image_indices >= 0
                label_non_image_indices = label_non_image_indices[valid_label_non_image_indices]

            # Create a mask for the image embedding positions.
            images_mask = torch.full(
                (batch_size, max_seq_len), True, dtype=torch.bool, device=input_ids.device
            )
            # No images in the text positions.
            images_mask[batch_indices, text_position_ids] = False
            # Samples can have different amount of images tokens. new_position_ids[:, -1] gives the last text position id for each sample.
            # Padding is needed when the number of image tokens differs.
            first_padding_idx = new_position_ids[:, -1] + 1
            images_mask[
                torch.arange(max_seq_len, device=first_padding_idx.device).repeat(batch_size, 1)
                >= first_padding_idx.unsqueeze(1)
            ] = False

        # Create the final input embedding (if this is the first language model stage).
        final_embedding = None
        if self.pre_process:
            embed_dim = language_embeddings.shape[-1]
            final_embedding = torch.zeros(
                batch_size,
                max_seq_len,
                embed_dim,
                dtype=image_embeddings.dtype,
                device=image_embeddings.device,
            )

            # Put text embeddings to the text positions in the result tensor.
            final_embedding[batch_indices, text_position_ids] = language_embeddings[
                batch_indices, non_image_indices
            ]

            # Put image embeddings to image positions.
            final_embedding[images_mask] = image_embeddings.reshape(-1, embed_dim).contiguous()

        # Create the final labels and loss mask (if this is the last language model stage).
        final_labels, final_loss_mask = None, None
        if has_labels:
            final_labels = torch.full(
                (batch_size, max_seq_len), IGNORE_INDEX, dtype=labels.dtype, device=labels.device
            )
            final_loss_mask = torch.full(
                (batch_size, max_seq_len), 0, dtype=loss_mask.dtype, device=loss_mask.device
            )

            # Put text labels and loss mask to the text positions.
            final_labels[label_batch_indices, label_text_position_ids] = labels[
                label_batch_indices, label_non_image_indices
            ]

            final_loss_mask[batch_indices, text_position_ids] = loss_mask[
                batch_indices, non_image_indices
            ]

            # For labels, we need to pick the last label index that got dropped by the shift to left.
            label_extra_text_position_ids = seq_lens - 1
            batch_range = torch.arange(len(label_extra_text_position_ids))
            final_labels[batch_range, label_extra_text_position_ids] = labels[batch_range, -1]

            # Loss mask the image positions.
            final_loss_mask[images_mask] = 0

            # Loss mask last text position just before an image so that text token does not need to predict the first image token.
            batch_image_indices, image_indices = torch.where(image_token_mask)
            # Indices just before image tokens. If it's -1, skip it.
            before_image_indices = image_indices - 1
            valid = before_image_indices >= 0
            valid_batch_image_indices = batch_image_indices[valid]
            valid_before_image_indices = before_image_indices[valid]
            # Map those indices those position ids.
            valid_before_image_indices = new_position_ids[
                valid_batch_image_indices, valid_before_image_indices
            ]

            final_loss_mask[valid_batch_image_indices, valid_before_image_indices] = 0

        if final_embedding is not None and has_labels:
            assert (
                final_embedding.shape[:2] == final_labels.shape == final_loss_mask.shape
            ), "unexpected shapes after data preprocessing"

        if final_embedding is not None:
            final_embedding = final_embedding.transpose(1, 0).contiguous()

        return final_embedding, final_labels, final_loss_mask
    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        fbank: Tensor = None,
        loss_mask: Optional[torch.Tensor] = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        image_token_index: Optional[int] = IMAGE_TOKEN_INDEX,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        #audio_embeddings = self.vision_projection(whisper_feats)
        #x=torch.load('fbank.pt')
        #current_device_index = torch.cuda.current_device()
        # 获取当前设备的对象
        #current_device = torch.device(f"cuda:{current_device_index}")
        #x.to(current_device)
        
        with torch.no_grad():
            
            #print(whisper_feats.unsqueeze(0).shape)
            whisper_feats = self.whisper_encoder(fbank.to(self.whisper_encoder.dtype))[0] 
            #print(x)
        
        audio_embeddings = self.vision_projection(whisper_feats)

        language_embeddings = self.language_model.embedding(
            input_ids=input_ids, position_ids=position_ids
        )
        audio_embeddings=audio_embeddings.permute(1, 0, 2)

        #pre_audio_emb = language_embeddings[:11]
        #suf_audio_emb = language_embeddings[11+ 1500:]
        #conbined_emb = torch.cat([pre_audio_emb, audio_embeddings, suf_audio_emb])
        
        #torch.save(conbined_emb, f'conbined_emb_{torch.distributed.get_rank()}')
        #torch.save(audio_embeddings, f'audio_embeddings_{torch.distributed.get_rank()}')
        #import time
        #time.sleep(5)

        # Assume 1 tile per image if the number of tiles is not provided.
        num_image_tiles=None
        if num_image_tiles is None:
            num_image_tiles = torch.ones(fbank.shape[0], dtype=torch.int, device=input_ids.device)
        use_inference_kv_cache = False
        language_embeddings = language_embeddings.transpose(
                1, 0
            ).contiguous()
        
        

        # Preprocess input, labels and loss mask.
        combined_embeddings, new_labels, new_loss_mask = self._preprocess_data(
            audio_embeddings,
            language_embeddings,
            input_ids,
            loss_mask,
            labels,
            use_inference_kv_cache,
            image_token_index,
            num_image_tiles,
        )  # [combined_seq_len, b, h_language], [b, combined_seq_len], [b, combined_seq_len]

        
        #input_ids=None,
        #position_ids=None,
        output = self.language_model(
            None,
            None,
            attention_mask,
            combined_embeddings,
            new_labels,
            inference_params,
            packed_seq_params,
            extra_block_kwargs)
        

            
        return output, new_loss_mask

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

