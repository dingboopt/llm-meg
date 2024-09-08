
# voicegpt
## prepare data
1. delete semantic from json field
```
python extract_asr.py asr.json asr_final_refined.json semantic
```
2. convert raw file to wds dataset format
```
python convert_voice_gpt_pretrain_to_wds.py . asr_final_refined.json wds_test
```
3. convert to energon format

    ```
    cd wds
    energon ./
    ```

    select the following values for the presented options:

    ```
    > Please enter a desired train/val/test split like "0.5, 0.2, 0.3" or "8,1,1": 9,1,0
    > Do you want to create a dataset.yaml interactively? [Y/n]: Y
    > Please enter a number to choose a class: 10 (VQAWebdataset)
    > Do you want to set a simple field_map[Y] (or write your own sample_loader [n])? [Y/n]: Y
    > Please enter a webdataset field name for 'image' (<class 'torch.Tensor'>): wav
    > Please enter a webdataset field name for 'context' (<class 'str'>): json
    > Please enter a webdataset field name for 'answers' (typing.Optional[typing.List[str]], default: None): 
    > Please enter a webdataset field name for 'answer_weights' (typing.Optional[torch.Tensor], default: None):
    ```
4.  Update  `pretrain_dataset.yaml`  so that both  `path`  variables point to  `wds`
 
5. convert hf ckt to meg:

convert from hf to mcore without projection weights:
```
python hf2mcore.py --load /workspace/sft/yc/yanc/voicegpt/exp/stage1/checkpoint-1000  --megatron-path ../../../ --load_path /workspace/sft/yc/yanc/voicegpt/exp/stage1/checkpoint-1000 --save_path  test  --target_params_dtype bf16 --target_tensor_model_parallel_size 4  --target_pipeline_model_parallel_size 1
```

convert from hf to mcore with projection weights:
```
python hf2mcoremm.py --load /workspace/sft/yc/yanc/voicegpt/exp/stage1/checkpoint-1000  --megatron-path ../../../ --load_path /workspace/sft/yc/yanc/voicegpt/exp/stage1/checkpoint-1000 --save_path  test  --target_params_dtype bf16 --target_tensor_model_parallel_size 4  --target_pipeline_model_parallel_size 1
```
7. start training:
```
bash run_voicegpt.sh xxx
```

8. convert from meg to hf:
```
python hf2mcore.py --load hf500/  --megatron-path ../../../ --load_path yyy/iter_0000002/ --save_path hf --target_params_dtype bf16 --target_tensor_model_parallel_size 4  --target_pipeline_model_par
allel_size 1 --convert_checkpoint_from_megatron_to_transformers
```

9. test
```
pip install cn2an  peft transformers==4.37.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=voicegpt/:$PYTHONPATH python  generate_from_wav.py --model hf
``` 

