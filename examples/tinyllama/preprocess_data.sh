#!/bin/bash

# 假设我们有一个数组
#array=(element1 element2 element3)

# 使用 seq 生成索引范围
for i in `seq 0 20`; do
    PYTHONPATH=../../:$PYTHONPATH python   ../../tools/preprocess_data.py --input /cloudfs-data/db/data/the_pile_deduplicated_jsonl/$i.jsonl    --output-prefix data/$i  --tokenizer-type HuggingFaceTokenizer  --tokenizer-model /cloudfs-data/db/model/TinyLlama_v1.1/   --append-eod --workers 16
    echo "Index: $i "
done
