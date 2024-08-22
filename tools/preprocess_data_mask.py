# coding=utf-8
# 数据预处理
# 同时可以输出loss_mask和att_mask的版本
# 原始数据类型(jsonlines)：
# [
#  {
#    "Q":"***", # question
#    "A":"***"  # answer
#   },
#   ...         # multi-round
# ]


import argparse
import json
import multiprocessing
import os
import sys
import random
from collections import defaultdict
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)))


import time

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.training.tokenizer import build_tokenizer
from megatron.core.datasets import indexed_dataset


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def encode_instruction(self, json_line):
        prefix = Encoder.tokenizer.tokenize("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:")
        human = Encoder.tokenizer.tokenize("\nHuman: ")
        ai = Encoder.tokenizer.tokenize("\nAssistant: ")
        sep = Encoder.tokenizer.tokenize("\n### Response:\n")

        data = json.loads(json_line)
        ids = {'text':[prefix], 'loss_mask':[[0] * len(prefix)]}
        #import pdb;pdb.set_trace()

        if len(data) == 1:
            que = Encoder.tokenizer.tokenize('\n'+data[0]['Q'])
            
            ans = Encoder.tokenizer.tokenize(data[0]['A'])
            tmp = que + sep
            ids['text'][0] += tmp
            ids['loss_mask'][0] += [0] * len(tmp)
            tmp = ans + [Encoder.tokenizer.eod]
            ids['text'][0] += tmp
            # eod predict next padding TODO: yanc&yz
            ids['loss_mask'][0] += [1] * len(tmp)
        else:
            for idx, line in enumerate(data):
                que = Encoder.tokenizer.tokenize(line['Q'])
                ans = Encoder.tokenizer.tokenize(line['A'])
                tmp = human + que + ai
                if idx == len(data)-1:
                    tmp += sep
                ids['text'][0] += tmp
                ids['loss_mask'][0] += [0] * len(tmp)
                ids['text'][0] += ans
                ids['loss_mask'][0] += [1] * len(ans)
            ids['text'][0] += [Encoder.tokenizer.eod]
            ids['loss_mask'][0] += [1]

        #print(Encoder.tokenizer.detokenize(ids['text'][0]))
        length = len(ids['text'][0])
        ids['att_mask'] = [[i for i in range(length)]]
        ids['len'] = [length]
        '''
        print(ids['text'][0])
        print(ids['loss_mask'][0])
        print(ids['att_mask'][0])
        print("----"*50)
        '''
        return ids, len(json_line)




def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, default='/disk2/yz/nlp/data/alpaca_gpt4_data_zh_new.jsonl',
                       help='Path to input JSON')
    group.add_argument('--output-prefix', type=str, default='/disk2/yz/tmp/data/cw/xxx',
                       help='Path to binary output file without suffix')
    group.add_argument("--tokenizer-model", type=str, default='/disk2/mnt/megatron_deepspeed/data/tokenizer',
                       help="Name or path of the huggingface tokenizer.")
    group.add_argument('--cat_as_padding', type=str, default='true', choices=['true', 'false'],
                       help='Whether to concat multiple samples into one sample.')
    group.add_argument('--cat_max_length', type=int, default=2048,
                       help='Only work when cat_as_padding is true. Length limit for concating multiple samples.')
    group.add_argument('--cat_epochs', type=int, default=3,
                       help='Only work when cat_as_padding is true. Number of epochs for concating multiple samples.')

    group.add_argument('--datasets', nargs='+', default=None,
                       help='Paths to one or more input datasets to merge')
    group.add_argument('--json-keys', nargs='+', default=['text','loss_mask', 'att_mask'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, default='PretrainedFromHF',
                       choices=['BertWordPieceLowerCase','BertWordPieceCase','LlamaTokenizer','HFAutoTokenizer',
                                'GPT2BPETokenizer', 'PretrainedFromHF', 'GPTSentencePieceTokenizer', 'HuggingFaceTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')

    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                            'This is added for computational efficieny reasons.')
    group.add_argument('--pad-vocab-size-to', type=int, default=250880,
                       help='Pad the vocab size to be divisible by this value.'
                            'Value of the size of the vocabulary of the tokenizer to reach. This value must be greater than'
                            ' the initial size of the tokenizer. If this argument is used the value of '
                            '`make-vocab-size-divisible-by` will be ignored.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=8,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    group.add_argument('--use-lossmask-as-weight', action='store_true')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True)

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    #encoded_docs = pool.imap(encoder.encode, fin, 25)
    encoded_docs = pool.imap(encoder.encode_instruction, fin, 25)

    level = "document"
    if args.split_sentences:
        level = "sentence"

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                    key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                    key, level)
        builders[key] = indexed_dataset.IndexedDatasetBuilder(output_bin_files[key],
        dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size))
        

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)
    print("args.json-keys: {}".format(args.json_keys))

    lens = defaultdict(int)
    if args.cat_as_padding == 'false':
        print('不把数据做拼接处理')
        time.sleep(1)
        for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            lens[doc['len'][0]] += 1
            for key in args.json_keys:
                sentences = doc[key]
                if len(sentences) == 0:
                    continue
                for sentence in sentences:
                    builders[key].add_item(torch.IntTensor(sentence))
                builders[key].end_document()
            if i % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed/elapsed/1024/1024
                print(f"Processed {i} documents",
                    f"({i/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)
    else:
        print(f'把数据做{args.cat_max_length}长度以内的拼接处理')
        time.sleep(1)
        all_samples = []
        for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            lens[doc['len'][0]] += 1
            all_samples.append(doc)
            if i % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed/elapsed/1024/1024
                print(f"Processed {i} documents",
                    f"({i/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)

        max_len = max(lens.keys())
        print(f'原始样本总数为：{sum(lens.values())}, 原始最大编码长度为：{max_len}')

        lens = defaultdict(int)
        print(f'共拼接生成{args.cat_epochs}轮的样本')
        
        def trancate_sample(text, loss_mask, att_mask):
            assert len(text) == len(loss_mask)
            if not args.use_lossmask_as_weight:
                return text, loss_mask, att_mask
            if len(text) > args.cat_max_length:
                print(f'Truncate sampel!!!')
                text = text[:args.cat_max_length ]
                text[-1] = tokenizer.eod
                assert len(text) == args.cat_max_length ,f'{len(text)}'
                loss_mask = loss_mask[:args.cat_max_length ]
                att_mask = att_mask[:args.cat_max_length ]
            # we need change loss mask to reweight
            loss_mask_sum = sum(loss_mask)
            if loss_mask_sum != 0:
                
                loss_mask = (np.array(loss_mask) * loss_mask_sum).tolist()
            
            return text, loss_mask, att_mask
            
            
        
        for _ in range(args.cat_epochs):
            random.shuffle(all_samples)
            text, loss_mask, att_mask = [], [], []
            for sample in all_samples:
                assert all(len(v) == 1 for v in sample.values())
                #import pdb;pdb.set_trace()
                if len(text) + sample['len'][0] > args.cat_max_length:
                    if text:
                        text, loss_mask, att_mask = trancate_sample(text, loss_mask, att_mask)
                        builders['text'].add_item(torch.IntTensor(text))
                        builders['loss_mask'].add_item(torch.IntTensor(loss_mask))
                        builders['att_mask'].add_item(torch.IntTensor(att_mask))
                        lens[len(text)] += 1
                        text = [] + sample['text'][0]
                        loss_mask = [] + sample['loss_mask'][0]
                        att_mask = [] + sample['att_mask'][0]
                        
                    else:
                        text, loss_mask, att_mask = trancate_sample(sample['text'][0], sample['loss_mask'][0], sample['att_mask'][0])
                        builders['text'].add_item(torch.IntTensor(text))
                        builders['loss_mask'].add_item(torch.IntTensor(loss_mask))
                        builders['att_mask'].add_item(torch.IntTensor(att_mask))
                        lens[len(text)] += 1
                        text, loss_mask, att_mask = [], [], []
                    builders['text'].end_document()
                    builders['loss_mask'].end_document()
                    builders['att_mask'].end_document()
                else:
                    text += sample['text'][0]
                    loss_mask += sample['loss_mask'][0]
                    att_mask += sample['att_mask'][0]
            if text:
                text, loss_mask, att_mask = trancate_sample(text, loss_mask, att_mask)
                builders['text'].add_item(torch.IntTensor(text))
                builders['loss_mask'].add_item(torch.IntTensor(loss_mask))
                builders['att_mask'].add_item(torch.IntTensor(att_mask))
                lens[len(text)] += 1
        builders['text'].end_document()
        builders['loss_mask'].end_document()
        builders['att_mask'].end_document()

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])

    max_len = max(lens.keys())
    print(f'样本总数为：{sum(lens.values())}, 最大编码长度为：{max_len}')
    total_cnt = sum(lens.values())
    sort_keys = sorted(lens.keys())
    cur = 0
    first_98 = first_99 = first_999 = first_9999 = True
    for k in sort_keys:
        cur += lens[k]
        if cur / total_cnt >= 0.98 and first_98:
            first_98 = False
            print(f'编码长度为{k}以内的数量超过了总量的98%')
        if cur / total_cnt >= 0.99 and first_99:
            first_99 = False
            print(f'编码长度为{k}以内的数量超过了总量的99%')
        if cur / total_cnt >= 0.999 and first_999:
            first_999 = False
            print(f'编码长度为{k}以内的数量超过了总量的99.9%')
        if cur / total_cnt >= 0.9999 and first_9999:
            first_9999 = False
            print(f'编码长度为{k}以内的数量超过了总量的99.99%')

if __name__ == '__main__':
    main()
