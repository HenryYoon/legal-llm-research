# Legal Lane + Solver

---

## 프로젝트 개요

작은 LLM(1.5B)에 법률 질문의 "추론 유형(Lane)"을 분류하게 가르치고, 논리 판정은 Z3 솔버에 맡겨서 7B급 CoT 모델을 이길 수 있는지 실험한다.

**핵심 가설**: Qwen2.5-1.5B + Lane + Z3 ≥ Qwen3-1.7B CoT.

**아이디어**: 법률 질문은 보통 (1) 사실 읽기 → (2) 요건 찾기 → (3) 요건 충족 여부 판정. 1.5B는 3에서 자주 엉키므로 3만 Z3에게 외주.

**Lane 5개** (R2 이후):

| Lane | 용도 | 처리 |
|------|------|------|
| L01 | 요건 매칭 | Z3 |
| L05 | 수치 계산 | SymPy |
| L06 | 설명·비교·요약 | 모델 직접 |
| L09 | 법률 용어 번역 | 모델 직접 |
| L10 | 정보 부족 | 모델 직접 |

**예시** (L01):
```
user:  "편의점에서 음료를 주머니에 넣고 나왔다. 절도죄?"

모델:  <lane>L01</lane>
       <tool_call>{
         elements: ["타인의 재물", "절취"],
         matching: {"타인의 재물": true, "절취": true}
       }</tool_call>

Z3:    {all_met: true}

모델:  "절도죄 성립. 두 요건 충족."
```

**환경**: Qwen2.5-1.5B-Instruct / Qwen3-1.7B, LegalBench 4개 태스크, RTX 3060 12GB.

> Lane 개념 출처: [DinoDS AI — Lanes](https://dinodsai.com/lanes). 본 연구는 법률 도메인에 맞춰 축소 재구성.

---

## 폴더 구조 및 사용법

### 구조

```
llm-research/
├── CLAUDE.md, README.md        루트 필수 2개
├── docs/plan.md                연구 계획 v4
├── context/                    Director → 연구원 지시 (R1/R2/R3)
├── solver/                     Z3 + SymPy 래퍼 + pytest
├── data/
│   ├── seeds*/                 Lane별 시드
│   ├── sft_*.jsonl, dpo_*.jsonl
│   └── legalbench/             평가 CSV
├── scripts/                    build / generate / validate / train / eval
├── configs/                    sft/dpo × qwen25/qwen3 × r1/r2/r3
├── reports/                    literature_review, data_quality
├── results/                    baseline, experiment_matrix, analysis
└── .claude/agents/             researcher1/2/3 정의
```

`models/`는 용량 문제로 git 제외.

### 사용법

```bash
# 환경 설정
uv venv .venv && source .venv/bin/activate
uv pip install transformers accelerate bitsandbytes peft trl datasets \
               z3-solver sympy jsonschema pyyaml pandas scikit-learn \
               rouge-score outlines

# 솔버 검증
python -m pytest solver/tests/

# 데이터 생성 (R3 기준)
python scripts/build_seeds.py
python scripts/generate_data_r3.py
python scripts/validate_data_r3.py

# Baseline 평가
python scripts/eval_baseline.py --model qwen25 --task all

# SFT 학습
python scripts/train_sft.py --config configs/sft_qwen25_r3.yaml

# Lane+Solver 평가 (A3)
python scripts/eval_with_solver_r3.py --model qwen25 --all-tasks \
       --adapter models/qwen25-1.5b-legal-sft-r3 \
       --max-samples 200 \
       --trace results/r3_a3_trace.jsonl \
       --output results/r3_a3.json
```

### 팀 구성 (Claude Code multi-agent)

| 역할 | 모델 | 담당 |
|------|------|------|
| Director | Opus 4.7 | 총괄, 리뷰, context 번역, 최종 분석 |
| 연구원 1 | Sonnet | 선행연구 조사 |
| 연구원 2 | Sonnet | 솔버 + 데이터 생성 |
| 연구원 3 | Haiku | 학습/평가 스크립트, 실행 |

연구원은 Agent tool로 일회성 spawn, 결과는 파일로 전달.

### 실험 매트릭스

|  | CoT만 | +Solver | +Lane+Solver |
|--|-------|---------|--------------|
| Qwen2.5-1.5B | A1 | A2 | **A3** |
| Qwen3-1.7B | B1 | B2 | B3 |
| Qwen3-1.7B thinking | C1 | — | — |

---

## 연구 진척

### 라운드 요약

| 라운드 | 주요 변경 | 결과 |
|--------|-----------|------|
| R1 | Lane 10개, 한국어 중심 SFT 19K | A3 mean **0.402**, 기각 [분석](results/analysis.md) |
| R2 | Lane 5개로 축소, 영어 70%, Outlines 제약 디코딩 | A3 mean **0.402** (변화 없음) [분석](results/analysis_r2.md) |
| R3 | Pipeline 파싱 버그 수정, 영문 시드 600+, binary regex 강제, schema enum | 🔄 SFT 진행 중 |

**목표 기준선**: A1 0.454, B1 0.412.

### 결과 (R1/R2, balanced-accuracy / ROUGE-L)

| Task | A1 | B1 | C1 | R1 A3 | R2 A3 |
|------|----|----|----|-------|-------|
| hearsay | 0.522 | 0.509 | 0.431 | 0.526 | 0.523 |
| personal_jurisdiction | 0.591 | 0.500 | 0.663 | 0.500 | 0.500 |
| rule_qa (ROUGE-L) | 0.121 | 0.139 | 0.113 | 0.092 | 0.117 |
| textualism_tool | 0.583 | 0.500 | 0.471 | 0.489 | 0.469 |
| **평균** | **0.454** | 0.412 | 0.419 | 0.402 | 0.402 |

### 주요 발견

- Qwen2.5-1.5B CoT가 Qwen3-1.7B CoT를 평균 +4pp 상회 — 주력 모델 선택 타당성 확인.
- Qwen3 thinking은 PJ(+16pp)에서만 유의미, 다른 태스크에서는 오히려 하락.
- A2가 A1을 이기는 태스크는 존재 (hearsay +2pp, rule_qa +5.5pp) — 솔버 효과 자체는 태스크 선별 시 유효.

### R2 실패 원인 (trace 검증)

1. **Pipeline 파싱 버그** — `<tool_call>...` 문자열이 final_answer로 누출, lane이 L06으로 오인됨.
2. **영문 시드 부족** — 한국어 SFT 과적합으로 영어 LegalBench에서 요건 환각(`element_of_purpose` 등).
3. **Binary 응답 포맷 부재** — Yes/No 물음에 paragraph 응답, parse_rate 28~35%.

R3에서 셋 모두 직접 공략 중.

### 코드 품질 (SOLID 점검)

R1~R3 솔버 코드가 병행 관리되어 중복 존재 (executor/validator/schemas 각 2~3중화). OCP·DIP 위반(하드코딩 dispatch). R3 평가 종료 후 `SolverProtocol` + Registry 패턴으로 리팩터 예정.

### 제약

- SFT 1 epoch: ~1시간 (transformers+PEFT, Unsloth 미사용).
- Qwen3 thinking 평가: CoT 대비 5배 느림.
- KLAC 한국어 데이터: 라이선스 명시 부재로 현재 미사용.

---

## 데이터 / 선행연구 / 라이선스

- **[LegalBench](https://huggingface.co/datasets/nguha/legalbench)** (CC-BY-4.0) — hearsay, personal_jurisdiction, rule_qa, textualism_tool_dictionaries.
- **[KLAC](https://huggingface.co/datasets/jihye-moon/klac_legal_aid_counseling)** — 한국어 법률 상담 QA, 현재 보류.
- **합성 데이터**: subagent가 Python으로 생성 (외부 API 없음).

**선행연구**:
- SatLM (NeurIPS 2023), Logic-LM (EMNLP 2023), NL2Logic (2026) — NL → Z3/FOL.
- SaulLM (NeurIPS 2024) — 540B 법률 토큰 pretrain, 파라미터 효율 비교군.
- DinoDS AI — Lanes ([링크](https://dinodsai.com/lanes)) — Lane 개념 원천.
- LegalBench 논문 (arXiv 2308.11462).

상세는 [`reports/literature_review.md`](reports/literature_review.md).

**라이선스**: 연구용 (학술 비상업). 데이터셋은 각 출처 라이선스 준수.

---

## 부록: 저장소 이전 이력

이 저장소는 `fb38ebc` 커밋(2026-04-19)으로 초기화. 그 이전 편집은 Claude Code 로컬 `~/.claude/file-history/`에 스냅샷만 있어 git으로 재구성 불가.

| 시기 | 내용 |
|------|------|
| 2026-04-08 ~ 17 | Bonsai 1-bit LLM + 솔버 탐색 |
| 2026-04-17 ~ 18 | 법률 도메인 pivot, plan.md v4 확정 |
| 2026-04-19 04:20 | 현재 세션 시작 |
| 2026-04-19 06:11 | `fb38ebc` 초기 commit |

### `.claude_old/` — 폐기된 multi-agent 시도

팀 기반 multi-agent(`bonsai-lane-solver`)를 먼저 시도했으나 30분 만에 토큰 소진. SendMessage 왕복 오버헤드가 실제 진전 대비 과도. 현재는 Agent tool 단발 호출 방식(`.claude/agents/`)으로 전환. `.claude_old/`는 학습용 보존, **실행 안 됨**.
