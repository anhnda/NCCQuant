python quantize.py --model-path ./models/Mistral-7B-v0.3 --quantizer nf4
python quantize.py --model-path ./models/Mistral-7B-v0.3 --quantizer nvfp4 --budget-p 0.03
python quantize.py --model-path ./models/Qwen2.5-7B --quantizer codebook3 --no-skip-lmhead

Llama 3.1. /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
Mistral 7B /home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1 
Qwen2.5 /home/DATA/prometheus/anh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796 

python quantize.py --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --quantizer nf4 --no-ncc --output-dir ./quantized_models/llama3.1_nf4



python quantize.py --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --quantizer nf4 --output-dir ./quantized_models/llama3.1_nf4_ncc

python quantize.py --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --quantizer nvfp4 --budget-p 0.03

python compare_slicing.py --heuristic-path ./quantized_models/llama3.1_nf4_ncc/ --standard-path ./quantized_models/llama3.1_nf4/

python compare_slicing.py --heuristic-path ./quantized_models/llama3.1_nvfp4/ 



Dataset         Heuristic AWQ (↓)  Standard AWQ (↓)   
--------------------------------------------------------------------------------
WikiText-2      5.8451          5.8737                   
C4              9.3916          9.4012                 



python quantize.py --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --quantizer nf4 --output-dir ./quantized_models/llama3.1_nf4_ncc
python compare_slicing.py --heuristic-path ./quantized_models/llama3.1_nf4_ncc/


python quantize.py --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --quantizer nf4 --no-ncc --output-dir ./quantized_models/llama3.1_nf4
python compare_slicing.py --heuristic-path ./quantized_models/llama3.1_nf4/
