# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain or SFT multimodal."""
from copy import deepcopy
from functools import partial
import os
import sys
import warnings

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))

from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from config import get_language_model_config, get_vision_model_config, get_vision_projection_config
from megatron.core.models.multimodal.llava_model import LLaVAModel
from layer_specs import get_layer_spec, get_mlp_module_spec, get_layer_spec_te
from megatron.training import pretrain
from dataloader_provider import train_valid_test_dataloaders_provider

import os
import torch
from functools import partial
from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.training import pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training.arguments import core_transformer_config_from_args


#from megatron_patch.model.qwen1_5.layer_specs import get_gpt_layer_with_transformer_engine_spec
#from megatron_patch.model.qwen1_5.model import GPTModel
#from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
#from megatron_patch.arguments import get_patch_args
from megatron.training import get_args
from megatron.training.tokenizer import build_tokenizer
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import logging
torch._logging.set_logs(dynamic=logging.ERROR, dynamo=logging.ERROR)

from voice_gpt import VoiceGpt

def model_provider(pre_process=True, post_process=True) -> Union[VoiceGpt, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    build_tokenizer(args)
    print_rank_0('building GPT model ...')
    num_image_tokens = args.audio_token_count

    old_seq_length = args.seq_length
    args.decoder_seq_length = args.seq_length + num_image_tokens
    args.seq_length = num_image_tokens
    if torch.distributed.get_rank() == 0:
        warnings.warn("Changed decoder_seq_length to num_image_tokens ({num_image_tokens}) + user-specified seq_length ({old_seq_length}).")

    if args.decoder_seq_length > args.max_position_embeddings:
        args.max_position_embeddings = args.decoder_seq_length
        warnings.warn("Expanded max_position_embeddings to {args.max_position_embeddings} to accommodate the full sequence of vit output + llm output.")
    # Experimental loading arguments from yaml
    config = core_transformer_config_from_args(args)
    adapter_list = args.adapter_list
   

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)

    model = VoiceGpt(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        adapter_list=adapter_list,
        encoder_path=args.encoder_path
    )
    model.freeze(freeze_language_model=True,  freeze_vision_projection=False)


    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    args = get_args()

    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None

    # Broadcast data.
    torch.cuda.nvtx.range_push("get_data")
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_text = tensor_parallel.broadcast_data(["text"], data, torch.int64)["text"]
    prompt_len = tensor_parallel.broadcast_data(["prompt_len"], data, torch.int64)["prompt_len"]
    target = tensor_parallel.broadcast_data(["target"], data, torch.int64)["target"]

    data_img = tensor_parallel.broadcast_data(["img"], data, torch.float32)

    torch.cuda.nvtx.range_pop()

    tokens_ = data_text.long()

    # 
    img_raw = data_img['img']


    torch.cuda.nvtx.range_push("index tokens")
    tokenizer = get_tokenizer()
    text_length = args.decoder_seq_length - args.seq_length
    tokens = tokens_[:, :text_length].contiguous()
    labels = tokens_[:, 1:text_length+1].contiguous()

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("get_ltor_masks_and_position_ids")
    if hasattr(tokenizer, 'eod'):
        eod_token = tokenizer.eod
    elif hasattr(tokenizer, 'eos_id'):
        eod_token = tokenizer.eos_id
    attention_mask, loss_mask, position_ids = \
        get_ltor_masks_and_position_ids(tokens, eod_token,
                                        args.reset_position_ids,
                                        args.reset_attention_mask,
                                        args.eod_mask_loss,
                                        question_length=prompt_len,
                                        target=target[:, 1:text_length+1]
                                        )
    torch.cuda.nvtx.range_pop()
    
    #if torch.distributed.get_rank() == 0:
    #    import pdb;pdb.set_trace()
    #else:
    #    import time;time.sleep(10000)

    return tokens, labels, loss_mask, attention_mask, position_ids, img_raw
def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    question_length=None,
                                    target=None,
                                    weights=None):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

     # Loss mask.
    if target != None: # use target to create loss mask that is created in data preparation step
        loss_mask = torch.ones(target.size(), dtype=torch.float, device=data.device)
        loss_mask[target == eod_token] = 0.0 # mask paddings
        loss_mask[target == -100] = 0.0 # mask prompts

    else: # default creation
        loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
        if eod_mask_loss:
            loss_mask[data == eod_token] = 0.0

        if question_length is not None:
            for b in range(micro_batch_size):
                loss_mask[b, :max(0, question_length[b].item() - 1)] = 0.0


    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()


    if question_length is not None:
        # Create a mask based on question_length
        question_length_mask = torch.arange(loss_mask.size(1), device=loss_mask.device)[None, :] < question_length[:, None]
        # Invert the mask (1 where we want to keep the loss, 0 where we want to zero it out)
        inverted_mask = ~question_length_mask
        # Apply the mask to loss_mask
        loss_mask = loss_mask * inverted_mask.float()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)
    if weights is not None:
        loss_mask = loss_mask * weights

    return attention_mask, loss_mask, position_ids
def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        
        # if prompt only, we got loss mask all zero, loss = 0 in this case
        loss = loss[0] / loss[1] if loss[1] else loss[0]
    else:
        if loss_mask.sum() == 0:
            loss = torch.sum(losses.view(-1) * loss_mask)
        else: 
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        #loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model: VoiceGpt):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, whisper_feats = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor, loss_mask = model(tokens, position_ids, attention_mask, labels=labels, fbank=whisper_feats, loss_mask=loss_mask)

    return output_tensor, partial(loss_func, loss_mask)

def add_multimodal_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title='multimodal arguments')
    group.add_argument('--valid-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--dataset-config', type=str, default=None)
    group.add_argument('--audio-token-count', type=int, default=3000, help='fbank seq len')
    group.add_argument('--encoder-path', type=str, default=None, help='audio encoder path')
    

    return parser

def is_dataset_built_on_rank():
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = build_tokenizer(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=args.data_path,
        blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
        split=args.split,
        path_to_cache=args.data_cache_path,
        mock=args.mock_data,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        with_loss_mask=args.with_loss_mask
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    """Build train, valid, and test datasets."""
    from data import MultiModalDataset
    args = get_args()
    assert len(args.data_path) == 1
    train_ds = MultiModalDataset(dir=args.data_path[0], padding_to=args.seq_length, pad_id=-1)
    return train_ds, None, None



if __name__ == "__main__":
    train_valid_test_dataloaders_provider.is_distributed = True

    pretrain(train_valid_test_dataloaders_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=add_multimodal_extra_args)
