# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import dataclasses
import json
import random
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

#import conversation as conversation_lib
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.transforms import Compose, RandAugment, RandomResizedCrop, Resize, ToPILImage

from megatron.core.models.multimodal.llava_model import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from megatron.energon import (
    Batch,
    CaptioningSample,
    DefaultTaskEncoder,
    OCRSample,
    SimilarityInterleavedSample,
    VQASample,
)
from megatron.energon.transforms import CustomTransform, MergeTransform
from megatron.training import get_args
from megatron.training.tokenizer import build_tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    
from transformers import WhisperProcessor

prompt_template = {
    "ASR_PROMPT":[
        ("<|startofaudio|>{sematic_tokens}<|endofaudio|>\n识别出音频内容:\n","{asr_result}"),
        ("<|startofaudio|>{sematic_tokens}<|endofaudio|>\n将上面音频内容转写出来:\n","{asr_result}"),
        ("<|startofaudio|>{sematic_tokens}<|endofaudio|>\n帮我把这个音频内容识别成文字:\n","{asr_result}"),
        ("识别出下面音频的内容:\n<|startofaudio|>{sematic_tokens}<|endofaudio|>\n","{asr_result}"),
        ("将以下音频转写成文字:\n<|startofaudio|>{sematic_tokens}<|endofaudio|>\n","{asr_result}"),
        ("下面音频内容是什么:\n<|startofaudio|>{sematic_tokens}<|endofaudio|>\n","{asr_result}"),
    ],
    "TTS_PROMPT":[
        ("将以下文本翻译成语音序列:\n{context}\n","<|startofaudio|>{sematic_tokens}<|endofaudio|>"),
        ("{context}\n将上面文本内容合成对应的语音\n","<|startofaudio|>{sematic_tokens}<|endofaudio|>"),
    ],
    "KWS_PROMPT":[
        ("<|startofaudio|>{sematic_tokens}<|endofaudio|>\n音频中是否包含关键词:{kws_list}\n", "{kws_result}"),
        ("以下音频中是否包含关键词：{kws_list}\n<|startofaudio|>{sematic_tokens}<|endofaudio|>\n", "{kws_result}")
    ],
    "PHONE_PROMPT":[
        ("将音素序列翻译成文字：\n<|startofphone|>{phone_tokens}<|endofphone|>\n", "{context}"),
        ("将文字翻译成音素序列：\n{context}\n","<|startofphone|>{phone_tokens}<|endofphone|>")
    ],
    "PRETRAIN_PROMPT":[
        ("<|startofaudio|>{sematic_tokens}<|endofaudio|>", ""),
    ]
}

# Type for intermediate batch, after batch()
@dataclass
class ImageTaskSample:
    __key__: str
    __subflavors__: Dict
    # (c, h, w)
    img: torch.Tensor
    text: np.ndarray
    prompt_len: np.int64
    target: torch.Tensor = None
    img_size: Optional[tuple] = None


# Typing for the resulting batch data after encode_batch()
@dataclass
class ImageTaskBatch(Batch):
    __keys__: List[str]
    __subflavors__: List[Dict]
    # (n, c, h, w)
    img: torch.Tensor
    # (n, seq_len)
    text: torch.Tensor
    # (n, 1)
    prompt_len: torch.Tensor
    # (n, seq_len)
    target: torch.Tensor

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Tokenizer:
    def __init__(self):

        args = get_args()
        self.args = args

        self.IMAGE_TOKEN_INDEX = -200
        self.initializer()

    def initializer(self):
        # Use Encoder class as a container for global data
        Tokenizer.tokenizer = build_tokenizer(self.args)
        if hasattr(Tokenizer.tokenizer, 'eod'):
            self.eod_token = Tokenizer.tokenizer.eod
        elif hasattr(Tokenizer.tokenizer, 'eos_id'):
            self.eod_token = Tokenizer.tokenizer.eos_id
        else:
            raise AttributeError('No eod token found in Tokenizer')
        self.split_token = 313131

        if (
            hasattr(self.args, "split_sentences") and self.args.split_sentences
        ):  # default false
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            library = "tokenizers/punkt/{}.pickle".format("english")
            # print("loading: " + library)
            splitter = nltk.load(library)
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Tokenizer.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params, lang_vars=CustomLanguageVars()
                )
            else:
                Tokenizer.splitter = splitter
        else:
            Tokenizer.splitter = IdentitySplitter()

    def __call__(self, text: str, padded: bool = True): # -> torch.Tensor:
        sentence = Tokenizer.splitter.tokenize(text)[0]
        sentence = Tokenizer.tokenizer.tokenize(sentence)
        return sentence

    def pad(self, content, seq_len=1024):
        out = np.pad(content, pad_width=(0,max(0,seq_len-len(content))), mode='constant', constant_values=self.eod_token)

        return out

class TaskEncoder(DefaultTaskEncoder[OCRSample, OCRSample, ImageTaskBatch, dict]):
    """A simple task encoder for captioning."""

    def __init__(
        self
    ):
        # Specify the batch_type for default batching (batching is performed here "manually" by
        # overwriting the `batch` method)
        super().__init__()

        self.args = get_args()

        self.tokenizer = Tokenizer()
        self.manual_prompts = prompt_template
        self.seq_len = self.args.decoder_seq_length - self.args.seq_length
        self.max_seq_len = self.seq_len

        self.txt_to_token_dict = {}
        encoder_path = self.args.encoder_path
        self.whisper_processor = WhisperProcessor.from_pretrained(encoder_path)



    def encode_sample(self, sample: Union[CaptioningSample, OCRSample, VQASample, SimilarityInterleavedSample]):
        if isinstance(sample, OCRSample):
            yield self.encode_ocr(sample)
        elif isinstance(sample, CaptioningSample):
            yield self.encode_captioning(sample)
        elif isinstance(sample, VQASample):
            is_llava_training = sample.__subflavors__['is_llava_training'] if 'is_llava_training' in sample.__subflavors__ else False

            if "llava" in sample.__key__ or is_llava_training:
                yield self.encode_llava_pretrain(sample)
            else:
                yield self.encode_vqa(sample)
        elif isinstance(sample, SimilarityInterleavedSample):
            if "llava" in sample.__key__:
                yield self.encode_llava_sft(sample)
            else:
                raise NotImplementedError('Sample format not supported')
        else:
            raise NotImplementedError('Sample format not supported')
    def encode_vqa(self, sample: VQASample):

        sample_augmentation = sample.__subflavors__['augmentation'] if 'augmentation' in sample.__subflavors__ else False

        waveform = sample.image
        fbank = self.whisper_processor(
                    waveform, sampling_rate=16000, return_tensors="pt"
                ).input_features[0]
        # randomly select a prompt
        prompt_idx = np.random.randint(len(prompt_template["ASR_PROMPT"]))
        cur_prompt = prompt_template["ASR_PROMPT"][prompt_idx]
        #print(f'########### prompt {cur_prompt}. context: {sample.context}')
        prompt_list = cur_prompt[0].split('{sematic_tokens}')
        assert len(prompt_list) == 2
        pre_audio = prompt_list[0]
        after_audio = prompt_list[1]
        pre_audio_token = self.tokenizer(pre_audio)
        after_audio_token = self.tokenizer(after_audio)
        
        


        answer_token = self.tokenizer(sample.context)
        # add IMAGE_TOKEN_INDEX
        prompt_len = len(pre_audio_token) + 1 + len(after_audio_token) 

        seq_len = self.max_seq_len + 4

        text_sample = np.concatenate([pre_audio_token, [IMAGE_TOKEN_INDEX], after_audio_token, answer_token])
        text_sample = self.tokenizer.pad(text_sample, seq_len)

        target = text_sample.copy()
        target[:max(0, prompt_len - 1)] = IGNORE_INDEX

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            img=fbank,
            text=text_sample,
            prompt_len=prompt_len,
            target=target
        )



    def batch(self, samples: List[ImageTaskSample]) -> ImageTaskBatch:
        batch = ImageTaskBatch(
            __keys__=[s.__key__ for s in samples],
            __subflavors__=[s.__subflavors__ for s in samples],
            img=torch.stack([s.img for s in samples]),
            text=torch.from_numpy(np.stack([s.text for s in samples], axis=0).astype(np.int64)),
            prompt_len=torch.from_numpy(np.array([s.prompt_len for s in samples], dtype=np.int64)),
            target=torch.from_numpy(np.stack([s.target for s in samples], axis=0).astype(np.int64)),
        )

        return batch

    def encode_batch(self, batch: ImageTaskBatch) -> dict:
        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]
        return raw


def print_error_handler(exc: Exception, key: Optional[str]):
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()

# From https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/mm_utils.py#L185
def tokenizer_image_token(args, prompt, tokenizer, has_image=True, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):

    if not has_image:
        input_ids = tokenizer(prompt)

    else:
        prompt_chunks = [tokenizer(chunk) for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0

        if args.tokenizer_type in ['Llama2Tokenizer', 'Llama3Tokenizer'] and len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')

    # # remove BOS token
    # if args.tokenizer_type in ['Llama2Tokenizer', 'Llama3Tokenizer']:
    #     return input_ids[1:]

    return input_ids
