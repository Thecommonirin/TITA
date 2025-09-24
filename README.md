# TITA: Token-Level Inference-Time Alignment for Vision-Language Models

## Quick Start
```bash
conda create -n tita python==3.10 -y
conda activate tita
pip install torch==2.0.1 torchvision==0.15.2
pip install -e .
```

## Benchmarking
```bash
tita-bench sample --out sample.jsonl
tita-bench run --api-key $KEY --model deepseek-vl2 \
  --questions sample.jsonl --images ./api_benchmark_results \
  --out results.jsonl
tita-bench eval --dir ./api_benchmark_results
```

## Dataset Layout
```
data/
├── texvqa/
│   └── train_images/
└── ocrvqa/
    └── images/
```

## Training
```bash
bash run_dpo.sh
```
For example:
```bash
deepspeed --include localhost:0,1,2,3 src/train_dpo_ours.py \
  --deepspeed src/configs/deepspeed/zero3_offload.json \
  --model_name_or_path bczhou/tiny-llava-v1-hf \
  ...
```


## Inference
```bash
python src/inference.py
python src/llava/serve/test_message.py --controller-address http://localhost:21001 \
  --model-name your-model-name --message "Describe the image in detail."
```


