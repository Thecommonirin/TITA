#!/usr/bin/env python3
"""
统一基准测试脚本：支持 API 与 本地模型 两种模式

API 模式依赖: tita-benchmarks (已在本仓库中提供)
本地模型模式依赖: torch, transformers, pillow, torchvision
"""

import argparse
import json
import os
import time
from typing import Optional, List, Dict


def iter_questions(questions_file: str) -> List[Dict]:
    with open(questions_file, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def run_api_mode(api_key: str, model: str, questions_file: str, image_folder: Optional[str], out_file: str, max_questions: Optional[int]):
    from tita_benchmarks.api.benchmark import APIModelBenchmark, ModelConfig

    model_map = {
        'deepseek-vl2': ('DeepSeek-VL2', 'deepseek-ai/deepseek-vl2'),
        'qwen2.5-vl': ('Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen2.5-VL-72B-Instruct'),
    }
    if model not in model_map:
        raise ValueError(f"Unsupported API model: {model}")
    name, model_id = model_map[model]

    cfg = ModelConfig(name=name, model_id=model_id, api_key=api_key)
    bench = APIModelBenchmark(cfg)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    bench.run(questions_file, image_folder, out_file, max_questions)
    print(f"API results saved to {out_file}")


def run_local_mode(hf_model_id: str, device: str, questions_file: str, image_folder: Optional[str], out_file: str, max_questions: Optional[int]):
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from PIL import Image

    processor = AutoProcessor.from_pretrained(hf_model_id)
    model = AutoModelForVision2Seq.from_pretrained(hf_model_id)
    model.to(device)
    model.eval()

    questions = iter_questions(questions_file)
    if max_questions:
        questions = questions[:max_questions]

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as fout:
        for q in questions:
            qid = q.get('question_id', '')
            text = q.get('text', '')
            image_rel = q.get('image', '')

            try:
                if image_rel and image_folder:
                    image_path = os.path.join(image_folder, image_rel)
                    image = Image.open(image_path).convert('RGB') if os.path.exists(image_path) else None
                else:
                    image = None

                t0 = time.time()
                if image is not None:
                    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
                else:
                    # 纯文本情况：部分 VLM 也支持无图输入
                    inputs = processor(text=text, return_tensors="pt").to(device)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                t1 = time.time()

                output_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                rec = {
                    "question_id": qid,
                    "model_name": hf_model_id,
                    "prompt": text,
                    "response": output_text,
                    "response_time": t1 - t0,
                    "success": True,
                }
            except Exception as e:
                rec = {
                    "question_id": qid,
                    "model_name": hf_model_id,
                    "prompt": text,
                    "response": "",
                    "response_time": 0.0,
                    "success": False,
                    "error_message": str(e),
                }

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Local model results saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description='Unified VLM benchmark (API or local model)')
    parser.add_argument('--mode', choices=['api', 'local'], required=True)

    # API mode
    parser.add_argument('--api-key', type=str)
    parser.add_argument('--api-model', type=str, choices=['deepseek-vl2', 'qwen2.5-vl'])

    # Local mode
    parser.add_argument('--hf-model-id', type=str)
    parser.add_argument('--device', type=str, default='cuda')

    # Shared
    parser.add_argument('--questions-file', type=str, required=True)
    parser.add_argument('--image-folder', type=str)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--max-questions', type=int)

    args = parser.parse_args()

    if args.mode == 'api':
        if not args.api_key or not args.api_model:
            raise SystemExit('--api-key and --api-model are required for API mode')
        run_api_mode(args.api_key, args.api_model, args.questions_file, args.image_folder, args.out, args.max_questions)
    else:
        if not args.hf_model_id:
            raise SystemExit('--hf-model-id is required for local mode')
        run_local_mode(args.hf_model_id, args.device, args.questions_file, args.image_folder, args.out, args.max_questions)


if __name__ == '__main__':
    main()


