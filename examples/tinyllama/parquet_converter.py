import glob
import os
import sys
from pathlib import Path
import pyarrow.parquet as pq
from tqdm import tqdm
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

def find_parquet_files(directory):
    # Recursively find all .parquet files within the specified directory
    pattern = os.path.join(directory, '**', '*.parquet')
    parquet_files = glob.glob(pattern, recursive=True)
    return parquet_files



def process(file, output_directory):
    directory_path = sys.argv[1] 
    output_directory = sys.argv[2]
    
    output_file = output_directory+'/'+file[len(directory_path):-len('parquet')]+'jsonl'
    #pds = pq.read_pandas(file, columns=None).to_pandas()
    parent_dir = Path(output_file).parent
    #if not os.path.exists(parent_dir):
    os.makedirs(parent_dir, exist_ok=True)
    pds = pq.read_pandas(file, columns=None).to_pandas()
    pds.to_json(path_or_buf=output_file, orient='records', lines=True, date_format='iso', date_unit='us')#, compression='gzip')
    #idx_file = output_file[:-len('.jsonl')]
    #cmd = f'PYTHONPATH=../../:$PYTHONPATH python   ../../tools/preprocess_data.py --input {output_file} --output-prefix {idx_file}  --tokenizer-type HuggingFaceTokenizer  --tokenizer-model /cloudfs-data/db/model/TinyLlama_v1.1/   --append-eod --workers 2'
    #return os.system(cmd)
    #return pds.to_json(orient='records', lines=True)
    #df = pd.read_json(output_file, lines=True)
    return file

def handle(parquet_files, i, output_directory):
    #process(parquet_files[0], output_directory)
    
    with ProcessPoolExecutor() as executor:
        # 提交所有文件的读取任务
        futures = [executor.submit(process, file, output_directory) for file in parquet_files]
    
        # 创建一个空的 DataFrame 用于合并结果
        #merged_df = pd.DataFrame()

        # 等待所有任务完成，并合并结果
        for future in as_completed(futures):
            #merged_df = pd.concat([merged_df, future.result()], ignore_index=True)

            future.result()
            #if future.result() !=0:
            #    print(f'cmd execute error num: {future.result()}')    
            #    raise


        #merged_df.to_json(os.path.join(output_directory, f'{i}.jsonl'), lines=True, orient='records')
#print('done')

#raise


    



if __name__=='__main__':
    # Usage example:
    directory_path = sys.argv[1] 
    parquet_files = sorted(find_parquet_files(directory_path))

    output_directory = sys.argv[2]

    total_len = len(parquet_files)

    batch_size = int(sys.argv[3])

    for i in tqdm(range((total_len+batch_size-1)//batch_size), desc="Processing"):
        start = i * batch_size
        end = min(i * batch_size + batch_size, total_len)
        
        #print(f'start:{start}. end:{end} !!!!!!')
        handle(parquet_files[start:end], i, output_directory)



   