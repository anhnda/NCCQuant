python quantize.py --model-path ./models/Mistral-7B-v0.3 --quantizer nf4
python quantize.py --model-path ./models/Mistral-7B-v0.3 --quantizer nvfp4 --budget-p 0.03
python quantize.py --model-path ./models/Qwen2.5-7B --quantizer codebook3 --no-skip-lmhead