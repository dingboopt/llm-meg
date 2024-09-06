from dataclasses import dataclass
from dataclasses import field
from functools import partial
import json
import os
import random
import re
from typing import Dict, List, Optional, Sequence
import cn2an
import jiwer
from local.speech_llama import SpeechLlamaForCausalLM
import soundfile as sf
import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor

from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from typing import Optional
#os.environ["TOKENIZERS_PARALLELISM"] = "false"  # tokenizer 的 warning，说会死锁

torch.manual_seed(42)
def rank0_print(*args):
    print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="/workspace/sft/yc-speech/yckj3851/models/llm/Yi/Yi-6B/"
    )
    whisper: str = field(default=True)
    freeze_llama: bool = field(default=True)
    freeze_adaptor: bool = field(default=False)
    max_length: Optional[int] = field(default=4096)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    test_path: str = field(default=None, metadata={"help": "Path to the test data."})
    asr_prompt_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    """
    These arguments vary depending on how many GPUs you have, \
    what their capacity and features are, and what size model you want to train.
    """
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. \
            You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "train batch size per device"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "eval batch size per device"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )

    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    num_train_epochs: Optional[float] = field(
        default=1,
        metadata={"help": "The number of training epochs."},
    )
    learning_rate: Optional[float] = field(
        default=2e-5, metadata={"help": "optimizer learning rate"}
    )

    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    weight_decay: Optional[float] = field(default=0.0)

    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to use gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="paged_adamw_32bit",
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )

    output_dir: Optional[str] = field(
        default="",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )

    save_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: float = field(
        default=100,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    logging_dir: Optional[str] = field(
        default=None, metadata={"help": "Tensorboard log dir."}
    )
    logging_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )

    report_to: Optional[str] = field(
        default="all",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    remove_unused_columns: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Remove columns not required by the model when using an nlp.Dataset."
        },
    )

class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        data_args: DataArguments,
        asr_prompt_file: str,
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.human_prompt_str = []

        with open(asr_prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            self.human_prompt_str.append(line.strip())

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
        # waveform = item["whisper_feat"]
        # while True:
        #     # 处理异常值
        #     if not os.path.exists(waveform):
        #         print(f"离线特征不存在： {waveform}.")
        #         return self.__getitem__(random.randint(0, len(self.list_data_dict) - 1))
        #     else:
        #         break

        ## 长度判断，如果总长度大于 3600，则忽略这个用例。其中，1500 是 whisper 特征的T长度。
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
    tokenizer: AutoTokenizer, data_args, whisper_encoder=None, whisper_processor=None
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args,
        asr_prompt_file=data_args.asr_prompt_path,
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


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    compute_dtype = torch.bfloat16
   

    # "auto" 是从配置文件中读取
    model = SpeechLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        #model_args=model_args,
        device_map="cpu",
        attn_implementation="flash_attention_2",
    )
    model = model.to(training_args.device)
    #model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    tokenizer.model_max_length = model_args.max_length

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

    # 固定LLM参数
    if model_args.freeze_llama:
        model.requires_grad_(False)
        model.model.eval()
        for p in model.get_speech_projector().parameters():
            p.requires_grad = True

    if model_args.freeze_adaptor:
        for p in model.get_speech_projector().parameters():
            p.requires_grad = False
    else:
        for p in model.get_speech_projector().parameters():
            p.requires_grad = True

    print_trainable_parameters(model)

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        whisper_encoder=whisper_encoder,
        whisper_processor=whisper_processor,
    )
    '''
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    '''
    local_rank = int(os.environ.get('LOCAL_RANK', -1))

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
   
    print(f"local_rank:{local_rank}, {[{i[0]:torch.mean(i[1].detach())} for i in trainer.model.get_model().speech_projector.named_parameters()]}")
    print(f"local_rank:{local_rank}, {trainer.get_train_dataloader()}")
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)  # 断点训练


if __name__ == "__main__":
    train()
