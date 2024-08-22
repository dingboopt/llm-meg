# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import argparse
import math
import json
import os
import sys
import soundfile as sf
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor



def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='heihei')

    group.add_argument('--local_rank', type=int, default=0,
        help='local rank passed from distributed launcher.')
    group.add_argument('--world_size', type=int, default=1,
        help='local rank passed from distributed launcher.')
    group.add_argument('--data_path', type=str, default='/workspace/sft/yc/yanc/voicegpt/data/dataset/stage2/train/dataset-20240201.json',
        help='local rank passed from distributed launcher.')
    group.add_argument('--sample_root', type=str, default='/cloudfs-data/db/data/voicegpt_total/',
        help='local rank passed from distributed launcher.')
    group.add_argument('--tokenizer_path', type=str, 
        help='local rank passed from distributed launcher.')
    group.add_argument('--encoder_path', type=str, default='/workspace/sft/yc-speech/models_hub/models/ssl/whisper/large-v3',
        help='local rank passed from distributed launcher.')

    args = parser.parse_args()

    return args




def main():
    args = get_args()

    data_path = args.data_path
    tokenizer_path = args.tokenizer_path
    list_data_dict = json.load(open(data_path, "r"))

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )
    encoder_path = args.encoder_path

    whisper_encoder = WhisperForConditionalGeneration.from_pretrained(
        encoder_path, device_map='auto', torch_dtype=torch.bfloat16
    ).get_encoder()
    whisper_encoder.requires_grad_(False)
    whisper_encoder.eval()
    whisper_processor = WhisperProcessor.from_pretrained(encoder_path)

    list_data_dict = json.load(open(data_path, "r"))

    #<bos><|Human|><|startofaudio|>[Speech特征]<|endofaudio|><|Token|>[文本prompt]<|Assistant|>[回答内容]<eos>

    
    local_rank = int(args.local_rank)
    world_size = int(args.world_size)
    
    total_len = len(list_data_dict)
    
    #total_len = 1000
    
    len_per_gpu = total_len//world_size
    resider = total_len - len_per_gpu * world_size
    
    def cal_start(local_rank, len_per_gpu, resider):
        return local_rank * len_per_gpu + (local_rank if local_rank < resider else resider)
    
    start = cal_start(local_rank, len_per_gpu, resider)
    end = cal_start(local_rank+1, len_per_gpu, resider) -1 
    print(f'{world_size}:{local_rank}-------{start} ---> {end}    total len{total_len}')
    
    
    
    
    for i in range(start, end+1):
        print(f'{world_size}:{local_rank} {start} ---> {end} : handling {i}')
        item = list_data_dict[i]
        task_token = item["id"].split("-")[0]
        prompt_str = ""
        sentence_str = ""
        before_speech_str = "<|Human|><|startofaudio|>"
        ## KWS 关键词任务
        if task_token == "<|kws|>":
            prompt_str = f"<|endofaudio|>{task_token}{kws_prompt[random.randint(0, len(kws_prompt)-1)]}{item['context']}\n<|Assistant|>"
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
            return __getitem__(random.randint(0, len(list_data_dict)))

        before_speech_id = [tokenizer.bos_token_id] + tokenizer(
            before_speech_str
        ).input_ids  # 添加 bos
        prompt_id = tokenizer(prompt_str).input_ids
        sentence_id = tokenizer(sentence_str).input_ids + [
            tokenizer.eos_token_id
        ]  # 在 target 后面加了 eos

        ### on-the-fly 提取特征
        wav_path = item["path"]
        #while True:
        # 处理异常值
        try:
            if wav_path != "":
                waveform, sr = sf.read(wav_path)
            else:
                waveform = None
                print(f"waveform is none")
                continue
            
        except Exception as e:
            print(f"Error occurred while processing {wav_path}.")
            continue

        ### 加载离线特征
        # wav_path = "None"  # 为了下面 3600 的判断，设置一个非空值
        # waveform = item["whisper_feat"]
        # while True:
        #     # 处理异常值
        #     if not os.path.exists(waveform):
        #         print(f"离线特征不存在： {waveform}.")
        #         return __getitem__(random.randint(0, len(list_data_dict) - 1))
        #     else:
        #         break

        ## 长度判断，如果总长度大于 3600，则忽略这个用例。其中，1500 是 whisper 特征的T长度。
        len_ids = len(before_speech_id) + len(prompt_id) + len(sentence_id)
        if (wav_path != "" and (len_ids >= 2100)) or (
            wav_path == "" and (len_ids >= 3600)
        ):
            print(f"\n[Warning] Sample {item['id']} size is large than 3600.")
            continue
    
        fbank = whisper_processor(
                    waveform, sampling_rate=16000, return_tensors="pt"
                ).input_features[0]
        
        with torch.no_grad():
            whisper_feats = whisper_encoder(fbank.unsqueeze(0).to(whisper_encoder.dtype)
            )[0]
        sample_dir_num = f'{args.sample_root}/{item["id"]}'
        if os.path.isdir(sample_dir_num):
            continue
        
        os.makedirs(sample_dir_num)
        torch.save(before_speech_id, f'{sample_dir_num}/before_speech_ids_cur')
        torch.save(prompt_id, f'{sample_dir_num}/prompt_ids_cur')
        torch.save(sentence_id, f'{sample_dir_num}/target_id_cur')
        torch.save(whisper_feats, f'{sample_dir_num}/whisper_feats')

    return before_speech_id, waveform, prompt_id, sentence_id




if __name__ == '__main__':

    main()

