import json
import os
import webdataset as wds

from tqdm import tqdm

import sys
llava_pretrain_dir = sys.argv[1]

# Paths to the dataset files
json_file = os.path.join(llava_pretrain_dir, sys.argv[2])
output = os.path.join(llava_pretrain_dir, sys.argv[3])

if not os.path.exists(output):
    os.mkdir(output)

# Load data
with open(json_file, 'r') as f:
    data = json.load(f)

with wds.ShardWriter(os.path.join(output, 'pretrain-%d.tar'), maxcount=2) as shard_writer:
    for entry in tqdm(data):
        with open(entry['wav_path'], "rb") as img_file:
                image_data = img_file.read()
        sample = {
            "__key__": entry['id'],
            "wav": image_data,
            "json": json.dumps(entry['text'], ensure_ascii=False).encode("utf-8"),
        }
        shard_writer.write(sample)

print(f"Dataset successfully converted to wds")
