import argparse
import json
import os
import time
import base64
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import tqdm

from openai import OpenAI


@dataclass
class ModelConfig:
    name: str
    model_id: str
    api_key: str
    base_url: str = "https://api.siliconflow.cn/v1"


@dataclass
class TestResult:
    question_id: str
    model_name: str
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    response_time: float
    success: bool
    error_message: Optional[str] = None


class APIModelBenchmark:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.results: List[TestResult] = []

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _vision_message(self, image_path: str, question: str) -> List[Dict]:
        b64 = self._encode_image(image_path)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ]

    def _text_message(self, question: str) -> List[Dict]:
        return [{"role": "user", "content": question}]

    def test_one(self, question_data: Dict, image_folder: Optional[str] = None) -> TestResult:
        qid = question_data.get("question_id", "unknown")
        text = question_data.get("text", "")
        image_rel = question_data.get("image", "")

        t0 = time.time()
        try:
            if image_rel and image_folder:
                img_path = os.path.join(image_folder, image_rel)
                messages = self._vision_message(img_path, text) if os.path.exists(img_path) else self._text_message(text)
            else:
                messages = self._text_message(text)

            resp = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                stream=False,
                temperature=0.1,
                max_tokens=1024,
            )
            t1 = time.time()

            return TestResult(
                question_id=qid,
                model_name=self.config.name,
                prompt=text,
                response=resp.choices[0].message.content,
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
                response_time=t1 - t0,
                success=True,
            )
        except Exception as e:
            t1 = time.time()
            return TestResult(
                question_id=qid,
                model_name=self.config.name,
                prompt=text,
                response="",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                response_time=t1 - t0,
                success=False,
                error_message=str(e),
            )

    def run(self, questions_file: str, image_folder: Optional[str], output_file: Optional[str], max_questions: Optional[int]) -> List[TestResult]:
        with open(questions_file, "r", encoding="utf-8") as f:
            questions = [json.loads(l) for l in f if l.strip()]
        if max_questions:
            questions = questions[:max_questions]

        self.results = []
        for q in tqdm.tqdm(questions, desc=f"Testing {self.config.name}"):
            self.results.append(self.test_one(q, image_folder))
            time.sleep(1)

        if output_file:
            self.save(output_file)
        return self.results

    def save(self, output_file: str) -> None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for r in self.results:
                f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")

    @staticmethod
    def create_sample(file_path: str, num_questions: int = 10) -> None:
        items = [
            {"question_id": f"sample_{i+1}", "text": "请描述这张图片中的内容。", "image": f"sample_{i+1}.jpg"}
            for i in range(num_questions)
        ] + [
            {"question_id": f"text_{i+1}", "text": "请解释什么是人工智能？", "image": ""}
            for i in range(5)
        ]
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")


