from abc import ABC
from abc import abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM
from transformers import LlamaModel
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/sft/yc-speech/yckj3851/models/llm/Yi/Yi-6B/",
    trust_remote_code=True,
)


class SpeechMetaModel:
    def __init__(self, config) -> None:
        super(SpeechMetaModel, self).__init__(config)
        # 方式一：映射到 LLM vocabulary token 上
        # self.speech_projector = nn.Linear(1024, config.vocab_size) # Hubert
        # self.speech_projector = nn.Linear(1280, config.vocab_size) # Whisper

        # 方式二：映射到 embedding 上
        # self.speech_projector = nn.Linear(1024, config.hidden_size) # hubert
        self.speech_projector = nn.Linear(1280, config.hidden_size)  # Whisper Hidden size = 1280
        # self.speech_projector = nn.Linear(1280, 80) # 80 之后做拼接
        
        #self.model_args = model_args

    def get_speech_projector(self):
        return self.speech_projector


class SpeechMetaForCausalLM(ABC):
    def __init__(self) -> None:
        super().__init__()

    # 抽象类。其他类继承该类必须实现所有抽象方法。
    @abstractmethod
    def get_model(self):
        pass

    def get_speech_projector(self):
        return self.get_model().get_speech_projector()

    def encode_speech(self, speech):
        # speech 特征有 Hubert 特征 (B,T,1024) 和 Whisper 特征（B,T,1280）可选
        # On-the-fly 时，直接传入audio音频， speech = self.get_model().hubert(speech)
        speech_tokens = self.get_speech_projector()(speech)

        # use head 80 dim， and mask the last 4016 dim， for SVD exp only
        # pad = torch.zeros(speech_tokens.shape[0], speech_tokens.shape[1], 4016).to(speech_tokens.device).to(speech_tokens.dtype)
        # speech_tokens = torch.cat((speech_tokens, pad), dim=2)
        return speech_tokens

    def mask_targets(labels):
        return labels

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        prompt_ids,
        fbank_feats,
        whisper_encoder,
    ):
        device = self.get_model().device
        dtype = self.get_model().dtype
        batch_size = len(fbank_feats)
        batch, target, batch_target_start, batch_target_len = [], [], [], []
        

            

        # 遍历 batch
        for i in range(batch_size):
            # 特征一：hubert
            # hubert_feat = torch.from_numpy(hubert[i]).to(device).to(dtype)
            # speech_feature = self.encode_speech(hubert_feat)

            # 特征二：whisper
            if fbank_feats[i] != None:

                ### on-the-fly 提特征
                with torch.no_grad():
                    whisper_feats = whisper_encoder(
                        fbank_feats[i].unsqueeze(0).to(whisper_encoder.dtype).to(device)
                    )[0]
                '''
                    # 如果固定住线性层，则其不参与梯度回传
                    if self.get_model().model_args.freeze_adaptor:
                        speech_feature = self.encode_speech(whisper_feats).squeeze(0)
                if not self.get_model().model_args.freeze_adaptor:
                    speech_feature = self.encode_speech(whisper_feats).squeeze(0)
                '''
                speech_feature = self.encode_speech(whisper_feats).squeeze(0)
                
                ### 离线加载特征
                # whisper_feats = torch.load(fbank_feats[i]).to(device).to(dtype)
                # speech_feature = self.encode_speech(whisper_feats).squeeze(0)

            else:
                # 纯文本 QA，就创建一个空的 tensor
                speech_feature = torch.empty(0).to(device).to(dtype)

            # 方式一：使用 gumbel_softmax 映射到 LLM vocabulary 64000 token 上
            # speech_feature_one_hot = nn.functional.gumbel_softmax(speech_feature, tau=1, hard=True)
            # speech_feature_word_embedding = torch.matmul(speech_feature_one_hot, self.get_model().embed_tokens.weight)

            # 方式二：使用 softmax 映射到 LLM vocabulary 64000 token 上，再与 llama-emb 权重相乘
            # speech_feature = nn.functional.softmax(speech_feature, -1)
            # speech_feature_word_embedding = torch.matmul(speech_feature, self.get_model().embed_tokens.weight)

            # 方式三：通过Linear layer 映射到 4096 embedding 上
            # speech_feature_word_embedding = speech_feature

            # print(f"[prompt_ids] {prompt_ids}")
            before_speech_ids_cur = prompt_ids["before_speech_id"][i]  # 添加 <bos>
            prompt_ids_cur = prompt_ids["prompt_ids"][i]
            target_id_cur = prompt_ids["sentence_ids"][i]
            

            
            # print(f"[before_speech_ids_cur]: {tokenizer.decode(before_speech_ids_cur)}\n[prompt_ids_cur]: {tokenizer.decode(prompt_ids_cur)}\n[target_id_cur]: {tokenizer.decode(target_id_cur)}\n")
            batch_target_start.append(
                len(before_speech_ids_cur)
                + speech_feature.shape[0]
                + len(prompt_ids_cur)
            )
            target.append(torch.tensor(target_id_cur, dtype=torch.long, device=device))

            # before 提取 emb
            before_speech_ids_cur = torch.tensor(
                before_speech_ids_cur, dtype=torch.long, device=device
            )
            before_speech_emb = self.get_model().embed_tokens(before_speech_ids_cur)
            # prompt 和 text 进行拼接，提取 emb
            input_text_id = torch.tensor(
                prompt_ids_cur + target_id_cur, dtype=torch.long, device=device
            )
            input_text_emb = self.get_model().embed_tokens(input_text_id)

            # print(f'[sample] {tokenizer.decode(before_speech_ids_cur)} <SpeechFeat> {speech_feature.shape} {tokenizer.decode(input_text_id)}')
            sample = torch.cat(
                (before_speech_emb, speech_feature, input_text_emb), dim=0
            )
            batch_target_len.append(sample.shape[0])
            batch.append(sample)

        new_inputs_embeds = pad_sequence(
            batch, batch_first=True, padding_value=tokenizer.pad_token_id
        )  # <PAD> 作为填充
        # new_attention_mask = new_inputs_embeds.ne(tokenizer.pad_token_id)
        new_labels = (
            torch.ones(
                batch_size, new_inputs_embeds.shape[1], dtype=torch.long, device=device
            )
            * -100
        )
        new_attention_mask = torch.zeros(
            batch_size, max(batch_target_len), dtype=torch.bool, device=device
        )
        new_position_ids = torch.zeros(
            batch_size, max(batch_target_len), dtype=torch.long, device=device
        )
        for i in range(batch_size):
            new_labels[i, batch_target_start[i] : batch_target_len[i]] = target[i]
            new_attention_mask[i, : batch_target_len[i]] = True
            new_position_ids[i, : batch_target_len[i]] = torch.arange(
                0, batch_target_len[i], dtype=new_position_ids.dtype, device=device
            )

        return (
            None,
            new_position_ids,
            new_attention_mask,
            past_key_values,
            new_inputs_embeds,
            new_labels,
            batch_target_start,
            batch_target_len,
        )


class SpeechLlamaModel(SpeechMetaModel, LlamaModel):
    def __init__(self, config) -> None:
        super(SpeechLlamaModel, self).__init__(config)


class SpeechLlamaForCausalLM(LlamaForCausalLM, SpeechMetaForCausalLM):
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = SpeechLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.debug_iter = 0

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        prompt_ids: Optional[torch.FloatTensor] = None,
        fbank_feats: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        whisper_encoder=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if fbank_feats is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                batch_target_start,
                batch_target_len,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                prompt_ids,
                fbank_feats,
                whisper_encoder,
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # if self.config.pretraining_tp > 1:
        #     lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        #     logits = [nn.functional.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        #     logits = torch.cat(logits, dim=-1)
        # else:
        #     logits = self.lm_head(hidden_states)
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # shift_logits = logits[..., :, :].contiguous()
            # shift_labels = labels[..., :].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            # print(f"[shift_logits] {shift_logits}")
            # print(f"[shift_logits shape] {shift_logits.shape}")
            # print(f"[shift_labels] {shift_labels}")
            # print(f"[shift_labels shape] {shift_labels.shape}")

            # FoR EXP : 额外尝试，添加 CTC Loss
            # target_ids, target_lens = [], []
            # for i in range(inputs_embeds.shape[0]):
            #     target_ids.append(torch.tensor(prompt_ids["sentence_ids"][i], dtype=torch.long, device=inputs_embeds.device))
            #     target_lens.append(torch.tensor(len(prompt_ids["sentence_ids"][i]), dtype=torch.long, device=inputs_embeds.device))
            # speech_feat_pad = pad_sequence(speech_feat_after_linear, batch_first=True, padding_value=tokenizer.pad_token_id).transpose(0, 1)
            # speech_len = torch.tensor([feat.shape[0] for feat in speech_feat_after_linear], device=speech_feat_pad.device, dtype=torch.long)
            # target_ids_pad = pad_sequence(target_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            # target_lens = torch.tensor(target_lens, device=speech_feat_pad.device, dtype=torch.long)

            # ctc_fct = nn.CTCLoss()
            # # log_probs 需要的 shape：(T, N, C)
            # loss_ctc = ctc_fct(speech_feat_pad.log_softmax(2).to(torch.float32), target_ids_pad, speech_len, target_lens)
            # # print(f"[CTC loss] {loss_ctc}")
            # loss = loss * 0.9 + loss_ctc * 0.1

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        res = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        return res

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        fbank_feats = kwargs.pop("fbank_feats", None)
        prompt_ids = kwargs.pop("prompt_ids", None)
        whisper_encoder = kwargs.pop("whisper_encoder", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        if fbank_feats is not None:
            _inputs["fbank_feats"] = fbank_feats
        if prompt_ids is not None:
            _inputs["prompt_ids"] = prompt_ids
        if whisper_encoder is not None:
            _inputs["whisper_encoder"] = whisper_encoder
        return _inputs
