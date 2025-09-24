# Repository Guidelines

## Project Structure & Module Organization
- `src/tita_benchmarks/api` holds the CLI logic; entry point `cli.py` registers as `tita-bench`.
- Training utilities live in `src/trainer` and `src/llava/...` for alignment and evaluation pipelines.
- Assets for docs/demos under `assets/`; benchmarking notebooks and scripts under `benchmark/`; config JSONs in `src/configs`. Keep datasets outside repo.

## Build, Test, and Development Commands
- `conda create -n tita python==3.10 -y && conda activate tita` prepares the expected environment.
- `pip install torch==2.0.1 torchvision==0.15.2` then `pip install -e .` installs dependencies and the editable package.
- `tita-bench sample --out ./api_benchmark_results/sample_questions.jsonl` seeds benchmark data; follow with `tita-bench run ...` and `tita-bench eval ...` to measure API endpoints.
- `python src/inference.py` runs local inference against the aligned checkpoint; `bash run_dpo.sh` is a reference for distributed training.

## Coding Style & Naming Conventions
- Use Python 3.10+ with 4-space indent, snake_case for functions/modules, PascalCase for classes, and uppercase constants.
- Follow PEP 8 and keep imports ordered by stdlib, third party, local. Add type hints for public interfaces and prefer docstrings describing I/O.
- No formatter config ships with the repo; run `black` (line length 88) or `ruff format` locally and lint with `ruff`/`flake8` before committing.

## Testing Guidelines
- No formal test suite ships with this snapshot; add `pytest` cases alongside modules or under `tests/` using `test_*.py` naming.
- For integration validation, reuse `tita-bench eval` outputs and `python src/llava/serve/test_message.py --controller-address ...` against a running worker.
- Document manual checks in PRs until automated coverage exceeds 70%; new modules should include at least smoke tests covering command-line parsing and core utilities.

## Commit & Pull Request Guidelines
- Repository bundle lacks git history; mirror the upstream style by writing imperative, present-tense summaries with optional Conventional Commit prefixes (`feat:`, `fix:`).
- Group related changes per commit and describe external effects or required follow-up in the body.
- PRs should link tracking issues, list reproduced commands, note dataset paths or secrets touched, and attach evaluation metrics or screenshots when UI components change.

## Security & Configuration Notes
- Never hard-code API keys; export `KEY` or other secrets before running CLI examples.
- Keep large datasets and checkpoints outside version control and reference their paths via config files under `src/configs`.
