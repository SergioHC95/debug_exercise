# Research Engineer Debug Exercise (CLI + Git)

Your task is to debug a minimal Python module in this repository. You must demonstrate basic Git workflow and run everything from the command line.

## Requirements
- Use the CLI to run the program.
- Clone the repo, create a feature branch, commit your fix(es), push your branch, and merge into `main`.
- Keep changes minimal and focused.

## Setup
```bash
git clone <REPO_URL>
cd debug_exercise

python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\activate # Windows PowerShell

# pip install -U pip  # only if pip < 21.3
pip install -e .
```

## Run
```bash
python3 -m debug_exercise --epochs 50 --lr 0.1
```