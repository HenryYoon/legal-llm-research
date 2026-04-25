# Legal LLM Reasoning with Lane Routing + Solver

Legal reasoning pipeline that combines **small LLM lane routing**, **constrained decoding**, and **Z3/SymPy solvers** for structured legal QA.

The goal is not to build a generic chatbot.  
This project tests whether a small LLM can delegate formal reasoning steps to external symbolic tools and outperform larger CoT-only baselines on selected legal reasoning tasks.

---

## Highlights

- Built a legal reasoning pipeline with **Qwen2.5-1.5B**, **Qwen3-1.7B**, **Z3**, **SymPy**, and **Outlines constrained decoding**
- Evaluated on selected **LegalBench** tasks: hearsay, personal jurisdiction, rule QA, textualism tool dictionaries
- Achieved **R3 A2 mean 0.433**, outperforming **Qwen3-1.7B CoT baseline 0.412**
- Improved hearsay balanced accuracy to **0.575**, above Qwen2.5-1.5B CoT baseline **0.522**
- Found that naive SFT can hurt tool-use behavior: A3 stayed at **0.402** due to weak element-fact matching after SFT
- Designed next-stage commercial roadmap: **RAG + citation grounding + solver validation + refusal handling**

---

## Core Idea

Legal QA often decomposes into three steps:

1. Read facts
2. Identify legal elements
3. Decide whether the facts satisfy those elements

Small LLMs often fail at step 3.  
This project delegates formal element matching and numerical reasoning to external tools.

```text
User Question
    ↓
Lane Classifier
    ↓
Tool Call Schema
    ↓
Z3 / SymPy Solver
    ↓
Constrained Final Answer
    ↓
LegalBench Evaluation
```

---

## Lane Design

| Lane | Purpose | Handler |
|---|---|---|
| L01 | Legal element matching | Z3 |
| L05 | Numeric calculation | SymPy |
| L06 | Explanation / comparison / summary | LLM |
| L09 | Legal term translation | LLM |
| L10 | Insufficient information / uncertainty | LLM |

Example:

```text
Question:
"Was the statement hearsay?"

Model:
<lane>L01</lane>
<tool_call>{
  "tool": "z3",
  "method": "check_elements",
  "elements": ["out_of_court_statement", "offered_for_truth"],
  "matching": {
    "out_of_court_statement": true,
    "offered_for_truth": true
  }
}</tool_call>

Solver:
{"all_met": true}

Final Answer:
"Yes. The statement satisfies both required hearsay elements..."
```

---

## Experiment Matrix

| Variant | Description |
|---|---|
| A1 | Qwen2.5-1.5B CoT only |
| A2 | Qwen2.5-1.5B + Solver |
| A3 | Qwen2.5-1.5B + Lane + Solver + SFT |
| B1 | Qwen3-1.7B CoT only |
| C1 | Qwen3-1.7B thinking mode |

---

## Results

| Task | A1 CoT | B1 Qwen3 CoT | R3 A2 Solver | R3 A3 SFT + Solver |
|---|---:|---:|---:|---:|
| hearsay | 0.522 | 0.509 | **0.575** | 0.516 |
| personal_jurisdiction | **0.591** | 0.500 | 0.511 | 0.500 |
| rule_qa / ROUGE-L | 0.121 | 0.139 | **0.159** | 0.134 |
| textualism_tool | **0.583** | 0.500 | 0.486 | 0.460 |
| **Mean** | **0.454** | 0.412 | **0.433** | 0.402 |

### Key Finding

The solver-only route, **A2**, outperformed the larger Qwen3-1.7B CoT baseline:

```text
R3 A2 mean: 0.433
Qwen3-1.7B CoT baseline: 0.412
Delta = +2.1pp
```

However, the SFT route, **A3**, did not improve performance.  
Trace analysis showed that the SFT model learned weak element-fact matching and often reduced effective tool use.

This is the main research takeaway:

> Solver augmentation is useful for formalizable legal tasks, but naive SFT can damage tool-use behavior unless the data distribution and schema supervision are tightly controlled.

---

## Technical Stack

- Models: Qwen2.5-1.5B-Instruct, Qwen3-1.7B
- Training: Transformers, PEFT, TRL
- Constrained decoding: Outlines
- Solvers: Z3, SymPy
- Evaluation: LegalBench, balanced accuracy, ROUGE-L
- Infra: RTX 3060 12GB, Python, pytest

---

## Project Structure

```text
.
├── solver/                  # Z3 / SymPy execution layer
├── scripts/                 # data generation, training, evaluation
├── configs/                 # SFT / DPO experiment configs
├── data/                    # seeds, synthetic data, LegalBench CSVs
├── reports/                 # literature review, data quality notes
├── results/                 # baseline and R1-R3 analysis
├── docs/                    # research plan and commercial roadmap
└── context/                 # multi-agent research instructions
```

---

## Quick Start

```bash
uv venv .venv
source .venv/bin/activate

uv pip install transformers accelerate bitsandbytes peft trl datasets \
  z3-solver sympy jsonschema pyyaml pandas scikit-learn rouge-score outlines
```

Run solver tests:

```bash
python -m pytest solver/tests/
```

Run baseline evaluation:

```bash
python scripts/eval_baseline.py --model qwen25 --task all
```

Run Lane + Solver evaluation:

```bash
python scripts/eval_with_solver_r3.py \
  --model qwen25 \
  --all-tasks \
  --max-samples 200 \
  --trace results/r3_a2_trace.jsonl \
  --output results/r3_a2.json
```

---

## Research Notes

Detailed analyses:

- [R3 Final Analysis](results/analysis_r3.md)
- [Commercial Roadmap](docs/roadmap_commercial.md)
- [Literature Review](reports/literature_review.md)

---

## Limitations

- This is a research PoC, not a production legal advice system.
- Results are limited to selected LegalBench tasks.
- SFT did not improve the solver path in R3.
- Korean legal data was not used due to license uncertainty.
- The next stage requires RAG, citation grounding, refusal handling, and 7B+ model evaluation.

---

## Next Direction

The next version will shift from PoC to production-oriented legal AI:

```text
Question
  → Legal RAG
  → Citation-grounded element extraction
  → Lane routing
  → Z3/SymPy validation
  → Refusal-aware final answer
  → Faithfulness / citation / latency evaluation
```

Target areas:

- Legal RAG
- Citation-grounded answer generation
- Tool-use SFT
- Refusal learning
- 7B+ model comparison
