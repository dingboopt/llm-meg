from dataclasses import dataclass
from dataclasses import field
from functools import partial
import json
import logging
import os
import random
import re
from typing import Dict, List, Optional, Sequence

import cn2an
import jiwer
from local.speech_llama import SpeechLlamaForCausalLM
from peft import get_peft_model
from peft import LoraConfig
from peft import prepare_model_for_int8_training
from peft import TaskType
import soundfile as sf
import torch
from torch import distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor

local_rank = int(os.environ.get('LOCAL_RANK', -1))
os.environ["WANDB_DISABLED"] = "true"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"  # tokenizer 的 warning，说会死锁


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/workspace/sft/yc-speech/yckj3851/models/llm/Yi/Yi-6B/"
    )
    whisper: str = field(default=True)
    freeze_llama: bool = field(default=True)
    freeze_adaptor: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    test_path: str = field(default=None, metadata={"help": "Path to the test data."})
    asr_prompt_file: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    kws_prompt_file: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    lora_enable: bool = False


class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        data_args: DataArguments,
        asr_prompt_file: str,
        kws_prompt_file: str,
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.kws_prompt = []
        with open(kws_prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            self.kws_prompt.append(line.strip())

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        <bos><|Human|><|startofaudio|>[Speech特征]<|endofaudio|><|Token|>[文本prompt]<|Assistant|>[回答内容]<eos>
        """
        item = self.list_data_dict[i]
        task_token = item["id"].split("-")[0]
        prompt_str = ""
        sentence_str = ""
        before_speech_str = "<|Human|><|startofaudio|>"
        ## KWS 关键词任务
        if task_token == "<|kws|>":
            prompt_str = f"<|endofaudio|>{task_token}{self.kws_prompt[random.randint(0, len(self.kws_prompt)-1)]}{item['context']}\n<|Assistant|>"
            if item["sentence"][0] == "|":
                # 没有关键词
                keywords_str = item["sentence"][1:].replace(",", "、")
                sentence_str = f"音频中不含有关键词{keywords_str}。"
            elif item["sentence"][-1] == "|":
                # 只有关键词
                keywords_str = item["sentence"][:-1].replace(",", "、")
                sentence_str = f"音频中含有关键词{keywords_str}。"
            else:
                k_true_str, k_false_str = item["sentence"].split("|")
                k_true_str = k_true_str.replace(",", "、")
                k_false_str = k_false_str.replace(",", "、")
                sentence_str = (
                    f"音频中含有关键词{k_true_str}，不含有关键词{k_false_str}。"
                )

        ## ASR 语音识别任务
        elif task_token == "<|asr|>":
            prompt_str = f"<|endofaudio|>{task_token}识别出音频内容\n<|Assistant|>"
            sentence_str = item["sentence"]
        ## QA 音频问答任务
        elif task_token == "<|qa|>":
            prompt_str = f"<|endofaudio|>{task_token}回答音频问题\n<|Assistant|>"
            sentence_str = item["sentence"]
        ## 文本的 QA 任务
        elif task_token == "<|text|>":
            prompt_str = f"<|endofaudio|>{task_token}{item['context']}\n<|Assistant|>"
            sentence_str = item["sentence"]
        ## ST 语音翻译任务（现在只实验纯中文到纯英文）
        elif task_token == "<|st|>":
            prompt_str = f"<|endofaudio|>{task_token}翻译音频内容为英文\n<|Assistant|>"
            sentence_str = item["sentence"]
        elif task_token == "<|tts|>":
            spkid = item["id"].split("-")[1]
            prompt_str = f"<|endofaudio|>{task_token}你是一个翻译工具，将给定的输入序列，转换成对应的数字序列，数字表示范围严格控制在0到127之间，用英文逗号隔开\n<|startofspeaker|>{spkid}<|startoftoken|>{item['context']}\n<|Assistant|>"
            sentence_str = item["sentence"]
        elif task_token == "<|sve|>":
            prompt_str = (
                f"<|endofaudio|>{task_token}两段音频是否为同一个说话人\n<|Assistant|>"
            )
            if item["sentence"] == "Y":
                sentence_str = f"两段音频为同一个说话人。"
            elif item["sentence"] == "N":
                sentence_str = f"两段音频不为同一个说话人。"
        else:
            print(f"[Error] The task of {item['id']} is not right!")
            return self.__getitem__(random.randint(0, len(self.list_data_dict)))

        before_speech_id = [self.tokenizer.bos_token_id] + self.tokenizer(
            before_speech_str
        ).input_ids  # 添加 bos
        prompt_id = self.tokenizer(prompt_str).input_ids
        sentence_id = self.tokenizer(sentence_str).input_ids + [
            self.tokenizer.eos_token_id
        ]  # 在 target 后面加了 eos

        ### on-the-fly 提取特征
        wav_path = item["path"]
        while True:
            # 处理异常值
            try:
                if wav_path != "":
                    waveform, sr = sf.read(wav_path)
                else:
                    waveform = None
                break
            except Exception as e:
                print(f"Error occurred while processing {wav_path}.")
                return self.__getitem__(random.randint(0, len(self.list_data_dict) - 1))

        ### 加载离线特征
        # wav_path = "None"  # 为了下面 3600 的判断，设置一个非空值

        if "<|text|>" in item["id"]:
            waveform = None

        ## 长度判断，如果总长度大于 3600，则忽略这个用例。
        # 1500 是 whisper 特征的长度。
        len_ids = len(before_speech_id) + len(prompt_id) + len(sentence_id)
        if (wav_path != "" and (len_ids >= 2100)) or (
            wav_path == "" and (len_ids >= 3600)
        ):
            print(f"\n[Warning] Sample {item['id']} size is large than 3600.")
            return self.__getitem__(random.randint(0, len(self.list_data_dict) - 1))

        return before_speech_id, waveform, prompt_id, sentence_id


def my_collator_fn(samples, whisper_encoder, whisper_processor):
    batch = {}
    before_speech_ids, fbank_feats, prompt_ids, sentence_ids = [], [], [], []
    for before_speech_id, waveform, prompt_id, sentence_id in samples:
        before_speech_ids.append(before_speech_id)
        ### on-the-fly 提取特征
        if waveform is None:
            fbank_feats.append(None)
        else:
            fbank = whisper_processor(
                waveform, sampling_rate=16000, return_tensors="pt"
            ).input_features[0]
            fbank_feats.append(fbank)

        ### 离线提特征
        # fbank_feats.append(waveform)

        prompt_ids.append(prompt_id)
        sentence_ids.append(sentence_id)

    batch["prompt_ids"] = {
        "before_speech_id": before_speech_ids,
        "prompt_ids": prompt_ids,
        "sentence_ids": sentence_ids,
    }
    batch["fbank_feats"] = fbank_feats
    batch["whisper_encoder"] = whisper_encoder
    return batch


def make_supervised_data_module(
    tokenizer: AutoTokenizer, data_args, whisper_encoder, whisper_processor
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        asr_prompt_file=data_args.asr_prompt_file,
        kws_prompt_file=data_args.kws_prompt_file,
    )
    # eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
    #                             data_path=data_args.test_path,
    #                             data_args=data_args,
    #                             asr_prompt_file=data_args.asr_prompt_path)
    data_collator = partial(
        my_collator_fn,
        whisper_encoder=whisper_encoder,
        whisper_processor=whisper_processor,
    )
    return dict(
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def compute_metrics(pred, tokenizer):
    pred = pred.cpu()
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    punctuation = "[,.!?，。、！？ ]"
    for i in range(len(pred_str)):
        prediction = pred_str[i].upper()
        prediction = cn2an.transform(prediction, "an2cn")
        prediction = re.sub(punctuation, "", prediction)
        pred_str[i] = prediction
        # label_str[i] = tokenizer._normalize(label_str[i]).upper()

    # cer = 100 * metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * jiwer.cer(label_str, pred_str)
    print(f"[CER] {cer}")
    return {"cer": cer}


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.local_rank = int(os.environ.get('LOCAL_RANK', -1))
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    # compute_dtype = torch.bfloat16

    print(f"training_args.device: {training_args.device}")

    ### 加载 Llama
    # "auto" 是从配置文件中读取

    rank0_print(f"[Loading] Llama.")
    model = SpeechLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map=training_args.device,
        # device_map="auto",
        torch_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    ### on-the-fly 提特征
    whisper_encoder = WhisperForConditionalGeneration.from_pretrained(
        model_args.whisper, device_map=training_args.device, torch_dtype=compute_dtype
    ).get_encoder()
    whisper_encoder.requires_grad_(False)
    whisper_encoder.eval()
    whisper_processor = WhisperProcessor.from_pretrained(model_args.whisper)

    ### 离线提特征
    # whisper_encoder = None
    # whisper_processor = None

    if training_args.lora_enable:
        print(f"[Log] Using LORA...")
        # Define LoRA Config
        if not model_args.freeze_adaptor:
            print(f"[Info] 训练线性层参数")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "speech_projector"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
        else:
            print(f"[Info] 固定线性层参数")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
        # prepare int-8 model for training
        model = prepare_model_for_int8_training(model)

        # add LoRA adaptor
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        whisper_encoder=whisper_encoder,
        whisper_processor=whisper_processor,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
        # compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
    )
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)  # 断点训练


if __name__ == "__main__":
    train()
