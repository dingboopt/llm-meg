#!/bin/bash
set -x
cp tiny/$1 -rf .
python hf2mcore.py --auto-remove-opt --load /cloudfs-data/db/model/TinyLlama_v1.1/ --megatron-path ../../ --load_path $1/ --save_path  hf$1  --target_params_dtype bf16 --target_tensor_model_parallel_size 1 --target_pipeline_model_parallel_size 1 --convert_checkpoint_from_megatron_to_transformers
python ../../tools//infer.py hf$1 'what is the color of sky?'