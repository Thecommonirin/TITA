# TITA: Token-Wise Inference-Time Alignment for Vision-Language Models

TITA 项目提供了一套用于视觉语言模型对齐与评估的工具链，包含离线训练脚本、推理入口以及 API 基准测评子包 `tita-benchmarks`。本指南帮助你快速搭建环境、运行基准和复现训练流程。

## 项目亮点 Highlights
- ✅ 一站式 CLI：`tita-bench` 支持生成样例题目、调用云端模型并自动汇总统计。
- ✅ 训练流水线完备：包含 DeepSpeed 配置、DPO 训练脚本与参考环境脚本。
- ✅ 推理/服务工具：提供本地推理脚本与 LLaVA 服务端工具，便于快速验证模型输出。

## 目录结构 Repository Layout
- `src/tita_benchmarks/api/`：API 基准 CLI 入口，`cli.py` 注册为 `tita-bench`。
- `src/trainer/`：奖励模型与对齐训练相关工具。
- `src/llava/`：LLaVA 推理、评测与服务端组件。
- `src/configs/deepspeed/`：最新的 DeepSpeed 配置文件集合。
- `benchmark/`：额外的基准脚本或分析工具。
- `assets/`：示例资源、可视化素材等。

## 环境搭建 Environment Setup
```bash
conda create -n tita python==3.10 -y
conda activate tita
pip install torch==2.0.1 torchvision==0.15.2
pip install -e .
```

## 运行 API 基准 Benchmark CLI
```bash
tita-bench sample --out ./api_benchmark_results/sample_questions.jsonl
DATE=$(date +%Y%m%d)
tita-bench run --api-key $KEY --model deepseek-vl2 \
  --questions ./api_benchmark_results/sample_questions.jsonl \
  --images ./api_benchmark_results \
  --out ./api_benchmark_results/deepseek_vl2_${DATE}.jsonl
tita-bench eval --dir ./api_benchmark_results
```
将云端模型 API 密钥保存在环境变量 `KEY` 中，避免硬编码。

## 数据集准备 Dataset Layout
```
data/
├── texvqa/
│   └── train_images/
└── ocrvqa/
    └── images/
```
请从官方渠道下载数据，并保持与上述目录结构一致。

## 使用 DeepSpeed 训练 Training with DeepSpeed
最新的 DeepSpeed 配置位于 `src/configs/deepspeed/`。建议使用根目录脚本启动训练：
```bash
bash run_dpo.sh
```
若需要手动执行，可参考下述指令（已同步更新配置路径）：
```bash
deepspeed --include localhost:0,1,2,3 src/train_dpo_ours.py \
  --deepspeed src/configs/deepspeed/zero3_offload.json \
  --model_name_or_path bczhou/tiny-llava-v1-hf \
  ...
```
在运行前请准备好奖励模型、预训练权重和 DPO 数据集路径。

## 推理与服务 Inference & Serving
```bash
python src/inference.py
python src/llava/serve/test_message.py --controller-address http://localhost:21001 \
  --model-name your-model-name --message "Describe the image in detail."
```
结合 `src/llava/serve/controller.py` 等组件，可部署完整的多模态服务流程。

## 评测 Evaluation
更多评测设置可参考 [LLaVA-1.5 文档](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)。结合 `tita-bench eval` 结果，可追踪 Token 消耗与响应延迟。

## 贡献 Contributing
协作者指南详见 `AGENTS.md`，包含代码风格、测试与提 PR 规范。欢迎通过 Issue/PR 反馈问题与改进建议。
