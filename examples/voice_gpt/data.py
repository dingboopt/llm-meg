import os

import sys

import torch





class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, dir, padding_to, pad_id=-1):
        self.dir=dir
        self.promt_start=None
        self.padding_to = padding_to
        self.pad_id = pad_id
        self.sampe_list=[]
        items=os.listdir(self.dir)
        self.len = len(items)
        for it in items:
            self.sampe_list.append(it)
        #print(self.sampe_list)
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        result = {}
        target_dir = f'{self.dir}/{self.sampe_list[idx]}'
        #target_dir='/workspace/sft/yc/yanc/voicegpt/db/'
        #print(target_dir)
        before_speech_ids_cur = torch.load(f'{target_dir}/before_speech_ids_cur',map_location='cpu')
        whisper_feats = torch.load(f'{target_dir}/whisper_feats',map_location='cpu')   
        prompt_ids_cur = torch.load(f'{target_dir}/prompt_ids_cur',map_location='cpu')
        target_id_cur = torch.load(f'{target_dir}/target_id_cur',map_location='cpu')
        # convert to tensor if neccessarily
        before_speech_ids_cur = torch.Tensor(before_speech_ids_cur)
        prompt_ids_cur = torch.Tensor(prompt_ids_cur)
        whisper_feats = torch.Tensor(whisper_feats)
        target_id_cur = torch.Tensor(target_id_cur)

        assert len(whisper_feats)==1
        if self.promt_start is None:
            self.promt_start = len(before_speech_ids_cur) + len(whisper_feats[0])
            assert self.promt_start <= self.padding_to
        else:
            assert len(before_speech_ids_cur) + len(whisper_feats[0]) == self.promt_start

        target_start = self.promt_start + len(prompt_ids_cur)
        total_len = target_start + len(target_id_cur)
        
        max_len = max(self.padding_to + 1, total_len)
        loss_mask = torch.zeros(max_len, dtype=torch.float32)
        loss_mask[target_start:total_len] = True
        text = torch.ones(max_len, dtype=torch.long)
        text = text * self.pad_id
        text[self.promt_start:total_len] = torch.cat((prompt_ids_cur, target_id_cur))
        text[:len(before_speech_ids_cur)] = before_speech_ids_cur
        
        # do truncation if neccessarily
        text = text[:self.padding_to + 1]
        loss_mask = loss_mask[:self.padding_to + 1]
        
        tokens = text[:self.padding_to]
        labels = text[1:self.padding_to+1]
        loss_mask = loss_mask[1:]
        
        position_ids = torch.arange(0, self.padding_to)
        

        #result['before_speech_ids_cur'] = before_speech_ids_cur
        result['whisper_feats'] = whisper_feats[0]
        result['tokens'] = tokens
        result['labels'] = labels
        result['loss_mask'] = loss_mask
        result['position_ids'] = position_ids

        return result

