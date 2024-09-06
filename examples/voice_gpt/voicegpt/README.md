# voice-gpt

# 1. VoiceGPT 介绍

​	VoiceGPT 是一个多模态语音大模型，可支持语音和文本数据的输入，输出为文本结果。

​	语音数据作为输入的功能包括：关键词检测、语音识别、语音问答、语音翻译。

​	文本数据作为输入的功能包括：文本问答。

# 2. 数据处理

## 2.1 数据集构建

使用 `local/build_dataset.py`脚本构建 json 格式数据，该脚本调用方式为：

```
python local/build_dataset.py --data dataset.txt --task asr
```

不同任务的 dataset.txt 格式不同，每列以换行符 "\t" 进行分隔，以下是不同任务每行的内容示例，以及最终生成到 json 文件的形式。

```python
### KWS
用例id	音频路径		待识别的关键词	真正出现的关键词
 {
  "id": "<|kws|>-AISHELLS0011W0299",
  "context": "创造,这也是,合理,他们也、时代，计在，因此",
  "sentence": "计在|创造,这也是,合理,他们也,时代,因此",
  "path": "/home/yckj3851/audio/Mandarin/Aishell/wav/train/S0011/BAC009S0011W0299.wav"
 },

### ASR
用例id	音频路径		音频文本
{
  "id": "<|asr|>-Y0000004207_ABTL18lIwOQ_S00437",
  "context": "",
  "sentence": "他需要一个在家里相夫教子",
  "path": "/home/yckj3851/audio/Mandarin/WenetSpeech/Y0000004207_ABTL18lIwOQ_S00437.wav"
},

### QA
用例id	音频路径		问题内容		回答内容
{
  "id": "<|qa|>-Baiduzhidao-0624322",
  "context": "小米note可以刷miui9吗",
  "sentence": "您好！小米Note可以刷MIUI 9，但是需要确认您的设备型号和当前安装的MIUI版本，以及确保您的设备已解锁引导程序。如果您不确定如何操作，请参考小米>官方网站或相关的论坛教程，以免造成设备损坏。",
  "path": "/home/yckj3851/audio/TTS/qa/MNBVC/QA/Baiduzhidao/0/Baiduzhidao-0624322.wav"
},

### TEXT
用例id	文本内容		文本回答
{
  "id": "<|text|>-Baiduzhidao-4946730",
  "context": "穿越火线枪战王者荒岛求生什么时候出",
  "sentence": "很抱歉，作为一个 AI 语言模型，我并不知道 \"穿越火线枪战王者荒岛求生\" 的具体情况和上市时间。不过，我建议您关注游戏的官方网站或社交媒体账号>，以获取最新消息和发布日期。希望您能快速找到您所需要的信息。",
  "path": ""
},

### TTS
用例id	音素序列		code_pt_文件路径
{
  "id": "<|tts|>-spk0040_male_2-003614",
  "path": "",
  "context": "0,26,7,35,44,34,33,24,35,40,46,31,24,35,44,45,10,39,7,4,44,27,10,38,39,12,23,42,45,16,15,19,41,10,18,4,42,47,10,39,7,35,43,30,24,35,42,45,25,38,13,19,41,34,33,14,40,45,25,13,41,34,33,7,4,40,45,10,39,7,35,44,21,14,40,48,0",
  "sentence": "119,119,119,54,14,14,14,14,14,47,77,77,10,109,109,109,92,80,80,70,46,73,5,5,7,109,109,92,92,80,37,95,95,37,5,5,63,18,80,80,50,84,45,47,77,10,74,74,89,89,84,72,45,45,58,1,65,65,88,26,89,94,42,76,125,125,86,86,24,111,111,84,84,72,45,45,1,86,86,36,36,36,36,36,61,61,27,27,27,49,49,49,49,49,54,54,54,14,14,14,45,47,43,77,10,10,109,92,92,78,78,78,113,109,109,92,92,70,122,23,91,91,91,59,8,3,3,85,46,73,5,26,70,23,106,118,59,59,59,30,30,85,46,47,43,10,74,74,89,84,72,72,45,45,47,43,77,10,109,109,109,109,92,98,98,98,115,105,105,105,90,90,57,49,103,56"
},

### ST
用例id	音频路径		音频文本		翻译文本
{
  "id": "<|st|>-WECHAT000003143",
  "context": "到了东莞长安给我打电话我去接你",
  "sentence": "Call me at Dongguan Changan when you arrive and I will come to pick you up.",
  "path": "/home/yckj3851/audio/Mandarin/WeChat/chdata.part1.wav.final/000003143.wav"
},
```

当每个任务以及数据集的 json 文件生成完毕后，可以使用下面脚本进行合并，只需在文件中配置需要组合的 json 文件即可。

```
python local/combine_json_file.py combine_data.json
```

## 2.1 whisper 离线特征提取

```
python local/extract_whisper_encoder.py
```

split_num 可以设定并发数量。

# 3. 训练脚本

训练包含两个阶段，并且两个阶段均固定住 whisper-encoder 的参数。

## 3.1 第一阶段

固定住 LLM，只训练线性层。执行 `./run-stage1.sh`

```python
#!/bin/bash

GPU_IDS="0,1,2,3,4,5,6,7"
METAROOT="/home/yckj3851/models/llm/Yi-6B/"		# 模型路径
DATAROOT="/home/yckj3851/immaculate/egs/llava/data/"	# 数据路径
WHISPER="/home/yckj3851/models/ssl/whisper/large-v3/"	# whisper 路径
PORT=29500	# deepspeed 端口号

DATA_PATH=${DATAROOT}/dataset/stage2/train/dataset-20240201.json	# 数据
ASR_PROMPT_PATH=${DATAROOT}/prompt/asr_prompt.txt		# ASR 任务 prompt 内容，暂时 prompt 都是固定的，所以没用到。
OUTPUT_PATH=exp/stage1		# 模型保存路径
ZERO_FILE=local/zero0.json

deepspeed --include localhost:${GPU_IDS} --master_port ${PORT} linear_pretrain.py \
    --deepspeed ${ZERO_FILE} \
    --model_name_or_path ${METAROOT} \
    --whisper ${WHISPER} \
    --data_path ${DATA_PATH} \
    --asr_prompt_path ${ASR_PROMPT_PATH} \
    --fp16 False \
    --bf16 True \
    --freeze_llama True \		# 固定 LLM 参数，只更新线性层
    --per_device_train_batch_size 1 \	# 每张卡上的 batchsize
    --gradient_accumulation_steps 16 \	# 多少个 batchsize 更新一次参数
    --dataloader_num_workers 0 \
    --learning_rate 1e-3 \		# 学习率
    --warmup_steps 1000 \
    --save_steps 1000 \
    --logging_steps 25 \
    --output_dir ${OUTPUT_PATH} \
    --max_steps 10000 \
    --greater_is_better False \
    --push_to_hub False 
```

在训练代码 `linear_pretrain.py` 中，对于 whisper 特征的提取进行了注释，可以在线提取，也可以离线提取。

## 3.2 第一阶段

固定住 线性层，使用 LoRA 微调 LLM。执行 `./run-stage2.sh`

```python
#!/bin/bash

GPU_IDS="0,1,2,3,4,5,6,7"
stage1=20240224
METAROOT="exp/stage1-${stage1}/checkpoint-20000/"
stage2=20240229
Freeze_Adaptor=True
dataset=dataset-20240224.json
OUTPUT_PATH=exp/stage2-${stage1}-${stage2}

DATAROOT="/home/yckj3851/immaculate/egs/llava/data/"
WHISPER="/home/yckj3851/models/ssl/whisper/large-v3/"

deepspeed --include localhost:${GPU_IDS} --master_port ${PORT} llama_finetune.py \
    --deepspeed ./local/zero0.json \
    --model_name_or_path ${METAROOT} \
    --whisper ${WHISPER} \
    --data_path ${DATAROOT}/dataset/stage2/train/${dataset} \
    --test_path ${DATAROOT}/aishell/test/dataset.json \
    --asr_prompt_file ${DATAROOT}/prompt/asr_prompt.txt \
    --kws_prompt_file ${DATAROOT}/prompt/kws_prompt.txt \
    --output_dir ${OUTPUT_PATH} \
    --fp16 False \
    --bf16 True \
    --freeze_llama False \
    --freeze_adaptor ${Freeze_Adaptor} \
    --lora_enable True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --dataloader_num_workers 0 \
    --learning_rate 1e-4 \
    --warmup_steps 300 \
    --save_steps 500 \
    --logging_steps 25 \
    --max_steps 3000 \
    --metric_for_best_model "cer" \
    --push_to_hub False 
```

同样，在训练代码 `llama_finetune.py` 中，对于 whisper 特征的提取进行了注释，可以在线提取，也可以离线提取。

## 3.3. 模型代码

参考 `local/speech_llama.py`