import glob
import os
import sys
from pathlib import Path

from tqdm import tqdm


def find_files(directory, suffix):
    # Recursively find all .parquet files within the specified directory
    pattern = os.path.join(directory, '**', f'*.{suffix}')
    #print(f'@@@ {directory}')
    files = glob.glob(pattern, recursive=True)
    return files






if __name__=='__main__':
    # Usage example:
    directory_path = sys.argv[1] 
    files = sorted(find_files(directory_path, 'jsonl'))
    begin = int(sys.argv[4])
    end = int(sys.argv[5])
    print(f'process file index : {begin}:{end} ')
    #print(files)
    if end==0:
        end = len(files)
    files = files[begin:end]
    output_directory = sys.argv[2]
    os.makedirs(output_directory, exist_ok=True)

    total_len = len(files)

    batch_size = int(sys.argv[3])

    for i in tqdm(range((total_len+batch_size-1)//batch_size), desc="Processing"):
        start = i * batch_size
        end = min(i * batch_size + batch_size, total_len)
        
        #print(f'start:{start}. end:{end} !!!!!!')
        #handle(parquet_files[start:end], i, output_directory)
        target_files = files[start:end]
        cmd = 'date &&'
        for path in target_files:
            file_name = os.path.basename(path)
            sub_cmd= f'PYTHONPATH=../../:$PYTHONPATH python ../../tools/preprocess_data.py --input {path} --output-prefix {output_directory}/{file_name[:-len(".json")]} --tokenizer-type HuggingFaceTokenizer --tokenizer-model /cloudfs-data/db/model/TinyLlama_v1.1/ --append-eod --workers 4 &'
            cmd = cmd + sub_cmd
        cmd = cmd + 'wait && date'
        #print(f'cmd is {cmd}')
        os.system(cmd)



   