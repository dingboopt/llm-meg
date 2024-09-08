import torch
import json
import argparse
import numpy as np
import datetime
import jiwer
import cn2an
import re
import soundfile as sf
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoTokenizer
from local.speech_llama import SpeechLlamaForCausalLM
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from peft import PeftModel

gpu_nums = 8
decode_batch_size = 32
compute_dtype = torch.float16
audio_path = "/workspace/sft/yc-speech/data/audio/Mandarin/Aishell/wav/test/S0764/BAC009S0764W0121.wav"
Whisper_Path = "/workspace/sft/yc-speech/models_hub/models/ssl/whisper/large-v3/"
prompt_str = "<|endofaudio|><|asr|>识别出音频中的内容\n<|Assistant|>"


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora-model", type=str, default="", required=False)
    # parser.add_argument("--data", type=str, default="", required=False)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    model = SpeechLlamaForCausalLM.from_pretrained(args.model, torch_dtype=compute_dtype, device_map="auto")
    if args.lora_model != "":
        model = PeftModel.from_pretrained(model, args.lora_model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
            args.model, 
            trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    whisper_encoder = WhisperForConditionalGeneration.from_pretrained(Whisper_Path, device_map="auto", torch_dtype='auto').get_encoder()
    whisper_encoder.eval()
    # whisper_encoder = whisper_encoder.to(device)
    whisper_processor = WhisperProcessor.from_pretrained(Whisper_Path)
    
    waveform, sr = sf.read(audio_path)
    fbank = whisper_processor(waveform, sampling_rate=sr, return_tensors="pt").input_features[0]
    with torch.no_grad():
        fbank = fbank.unsqueeze(0).to(whisper_encoder.device).to(whisper_encoder.dtype)
        feature = whisper_encoder(fbank)[0].squeeze(0) #(T, C)
    
    
    before_speech_ids = [tokenizer.bos_token_id] + tokenizer("<|Human|><|startofaudio|>").input_ids
    prompt_ids = tokenizer(prompt_str).input_ids  # KWS
    before_speech_embs = model.get_model().embed_tokens(torch.tensor(before_speech_ids, dtype=torch.long).to(model.device))
    prompt_embs = model.get_model().embed_tokens(torch.tensor(prompt_ids, dtype=torch.long).to(model.device))
    print(f"[Prompt] {prompt_str}")
    
    with torch.no_grad():
        speech_emb = model.encode_speech(feature.to(model.device))
        con_embeds = torch.cat([before_speech_embs, speech_emb, prompt_embs])
        generate_input = {
                    "inputs_embeds":con_embeds.unsqueeze(0),
                    "max_new_tokens":100,
                    "do_sample":True,
                    "top_k":50,
                    "top_p":0.95,
                    "temperature":0.3,
                    "repetition_penalty":1.3,
                    "eos_token_id":tokenizer.eos_token_id,
                    "bos_token_id":tokenizer.bos_token_id,
                    "pad_token_id":tokenizer.pad_token_id
                }
        generate_ids = model.generate(**generate_input)
        batch_res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        print(batch_res)
        
    

    
if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    main()
