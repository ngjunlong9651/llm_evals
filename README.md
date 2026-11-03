# LLM Evals: learn by measuring

Start from the failures you observe, validate your judge, and put error bars on every number. All data is **synthetic with known ground truth**, so you can check that each measurement method recovers the truth.

## Contents
| File | What it is |
|---|---|
| `evals_learning.ipynb` | The lecture notebook (concepts, labs, checkpoints, exercises, glossary, reading list) |
| `evals.py` | Graders, a simulated LLM judge, bootstrap CIs, Cohen's kappa, judge bias correction |
| `requirements.txt` | numpy, pandas, matplotlib, ipykernel |

Runs **offline**, no API keys.

## Slides
A companion slide deck lives on the author's Claude account (private link, may not open for you): https://claude.ai/artifact/Rvbz1Na7RMnTUU8ZBRGmx8

## Setup
```bash
uv venv .venv
uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name llm-evals --display-name "Python (llm-evals)"
```
Open the notebook and select the **Python (llm-evals)** kernel.

## Topics
Error analysis and Pareto charts, code-based graders, LLM-as-judge design, validating a judge (TPR, TNR, kappa), correcting judge bias, bootstrap intervals, paired comparisons, regression suites and common traps.

## Series
Part of a small AI engineering learning series: Hybrid RAG, Harness Engineering, LLM Evals, Tool Use and MCP.
