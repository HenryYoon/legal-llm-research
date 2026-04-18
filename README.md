# Legal Lane + Solver Research

> **작은 모델(1.5B)에 법률 추론을 "어떤 방식으로 생각해야 하는지"의 맵을 가르치고,
> 형식 논리로 풀 수 있는 부분은 Z3 솔버에게 넘겨,
> 훨씬 큰 모델(7B급 CoT)을 법률 벤치마크에서 이길 수 있는가?**

한 줄 요약: 소형 LLM + **Lane 분류** + **외부 솔버(Z3/SymPy)** 하이브리드가 파라미터 수로는 불리한 싸움을 뒤집을 수 있는지 실험하는 연구 저장소입니다.

- 베이스 모델: **Qwen2.5-1.5B-Instruct** (주력), Qwen3-1.7B (비교군)
- 평가: **LegalBench** 4개 태스크 + 한국어 법률 상담 QA
- 하드웨어: **RTX 3060 12GB** 단일 GPU

---

## 목차

1. [왜 이 연구인가?](#왜-이-연구인가)
2. [Lane이란 무엇인가?](#lane이란-무엇인가)
3. [아키텍처](#아키텍처)
4. [실험 디자인](#실험-디자인)
5. [현재 진행 상태](#현재-진행-상태)
6. [결과 요약](#결과-요약)
7. [디렉토리 구조](#디렉토리-구조)
8. [실행 방법](#실행-방법)
9. [팀 구성 (Claude Code multi-agent)](#팀-구성-claude-code-multi-agent)
10. [알려진 제약 및 이슈](#알려진-제약-및-이슈)
11. [데이터 / 선행연구 / 라이선스](#데이터--선행연구--라이선스)
12. [부록](#부록)

---

## 왜 이 연구인가?

### 배경 — 법률 추론은 작은 LLM에게 왜 어려운가

법률 질문("이 행위가 절도죄에 해당하는가?")에 올바로 답하려면 세 단계가 필요합니다.

1. **사실관계 파싱** — 자연어에서 "누가 무엇을 했는가"를 뽑아내기.
2. **적용 법률/요건 식별** — 예: 형법 329조 절도죄 요건(타인의 재물 + 절취).
3. **요건-사실 대응 판단** — 각 요건이 실제로 충족되었는가를 논리적으로 결정.

1단계·2단계는 언어 이해/암기 영역이지만, **3단계는 형식 논리(Boolean AND/OR, 조건부)**입니다. 그런데 작은 LLM은 체계적 논리 추론이 약합니다. 큰 모델(7B+)은 Chain-of-Thought(CoT)로 "단계별로 풀자"를 학습해 3단계까지 어느 정도 커버하지만, **1.5B급은 CoT로도 일관성이 무너지는 경우가 잦습니다.**

### 가설 — Lane + 솔버

위 3단계를 똑같이 LLM 안에서 전부 처리하는 대신:

- 작은 모델은 **"이 질문은 어떤 유형의 추론인가"(Lane)** 를 먼저 분류하고,
- Lane이 "형식 논리형"이면 **요건·사실을 JSON으로 정리만** 해서 Z3 솔버에게 **외주**하고,
- Lane이 "설명/요약형"이면 그냥 텍스트로 답합니다.

분류와 JSON 포매팅은 작은 모델이 잘하는 일이고, 논리 판단은 Z3가 확정적으로 잘하는 일. **작업을 쪼개서 각자 잘하는 쪽에 맡기자**는 가설입니다.

### 왜 법률 도메인인가

- 법률 추론은 연역(deductive) 비율이 높아 **형식화 여지가 큼**.
- "요건이 충족되었는가"의 구조가 Boolean AND와 거의 1:1 대응됨.
- **LegalBench**(162 태스크)라는 벤치마크가 존재해 정량 평가 가능.
- 범용 수학(GSM8K 등)보다 Lane 경계(형식 vs 생성)가 더 뚜렷함.

---

## Lane이란 무엇인가?

Lane은 **"이 질문에 답하려면 어떤 종류의 추론이 필요한가"** 를 분류하는 태그입니다. 모델이 질문을 받자마자 `<lane>L01</lane>` 같은 태그를 먼저 출력하게 SFT로 학습시킵니다.

### R2(현재) Lane 체계 — 5개

| Lane | 의미 | 처리 | 예시 |
|------|------|------|------|
| **L01 요건 매칭** | 법적 요건 vs 사실관계 Boolean 매칭 | **Z3** | "편의점 음료를 주머니에 넣고 나왔다 → 절도죄 성립?" |
| **L05 계산** | 수치 계산 | **SymPy** | "원금 1,000만원, 연 5%, 3년 이자?" |
| **L06 설명/비교/요약** | 자유 서술 | 모델 직접 | "가집행과 확정판결의 차이는?" |
| **L09 번역** | 법률 용어 한↔영 | 모델 직접 | "hearsay를 한국어로?" |
| **L10 불확실** | 정보 부족 | 모델 직접 | "이 상황에서 과실 비율은?" (사실 부족) |

> **R1**에서는 Lane 10개(L01~L10)로 시작했으나, 실험 결과 L02/L03/L04가 L01과 실무적으로 구분이 모호하고 분류 노이즈만 발생해 **R2에서 5개로 축소**. ([상세](results/analysis.md))

### 예시 — 같은 질문이 Lane에 따라 어떻게 처리되는가

**L01 (Z3 외주):**
```
user:      "피고인이 편의점에서 음료를 주머니에 넣고 나왔다. 절도죄 성립?"

model:     <lane>L01</lane>
           <tool_call>{
             "tool": "z3", "method": "check_elements",
             "elements": ["타인의 재물", "절취"],
             "facts":    ["편의점 소유 음료", "주머니에 넣고 퇴장"],
             "matching": {"타인의 재물": true, "절취": true},
             "mode": "and"
           }</tool_call>

tool:      {"all_met": true}   ← Z3가 확정적으로 판정

model:     "절도죄(형법 329조) 성립. 음료는 '타인의 재물'에 해당하고
            주머니에 넣고 퇴장하는 행위는 '절취'에 해당함. 두 요건이
            모두 충족되어 구성요건 성립."
```

**L06 (직접 생성):**
```
user:      "가집행과 확정판결의 차이는?"

model:     <lane>L06</lane>
           가집행은 판결이 아직 확정되지 않은 상태(상소 가능)에서
           이루어지는 집행을 말하고, 확정판결은 불복수단이 모두 소진되어
           더 이상 다툴 수 없게 된 판결의 집행을 의미합니다. ...
```

핵심: **작은 모델은 "Lane 분류 + JSON 포매팅"만** 잘하면 됩니다. 논리 연산은 Z3가 대신 합니다.

---

## 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│         Qwen 1.5~1.7B (LoRA SFT + DPO 적용)              │
│                                                          │
│   입력 질문 ──▶ Lane 분류 ──▶ 분기                      │
│                                │                         │
│                   ┌────────────┼─────────────┐          │
│                   ▼            ▼             ▼          │
│              L01/L05        L06/L09        L10          │
│           (tool_call JSON)  (직접 생성)   (불확실 답)     │
└───────────────────┬────────────────────────────────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   ┌───────────┐       ┌───────────┐
   │    Z3     │       │   SymPy   │
   │ (요건 매칭)│       │  (계산)   │
   └───────────┘       └───────────┘
          │                   │
          └─────────┬─────────┘
                    ▼
              tool 결과
                    │
                    ▼
          모델이 최종 자연어 답변 생성
```

**R2 핵심 기법:**
- **Outlines** 제약 디코딩: tool_call JSON이 반드시 schema를 따르도록 강제 (R1의 "환각 메서드" 문제 해결).
- **LoRA SFT**: 9K 샘플 (영어 77.5% + 한국어 22.5%), final answer 평균 639자.
- **DPO**: 10K 쌍, `rejected`에 schema 밖 메서드 / 짧은 답변 케이스 포함.

---

## 실험 디자인

### 실험 매트릭스

|  | CoT only (기저) | + Solver (Lane 없이) | + Lane + Solver (완전형) |
|--|-----------------|----------------------|--------------------------|
| **Qwen2.5-1.5B** | A1 | A2 | **A3** |
| **Qwen3-1.7B** | B1 | B2 | B3 |
| **Qwen3-1.7B thinking** | C1 | — | — |

### 검증 가설

- **A2 > A1** → 법률 추론에 솔버가 실제로 도움이 됨
- **A3 > A2** → Lane 분류가 솔버 효과를 증폭함
- **A3 > B1** → **핵심 가설**: 1.5B + Lane + Solver가 1.7B CoT보다 높음
- **A3 vs C1** → 외부 솔버 vs 내부 thinking 중 어느 쪽이 법률 추론에 더 효율적인가

---

## 현재 진행 상태

프로젝트는 **R1(실패 + 원인 분석) → R2(재설계 + 재학습)** 순으로 진행 중입니다.

### R1 결과 (완료)

| Step | 내용 | 상태 |
|------|------|------|
| 1 | 선행연구 조사 | ✅ [`reports/literature_review.md`](reports/literature_review.md) |
| 2 | Director 리뷰 → context/researcher2 | ✅ Lane 재배치 확정 |
| 3 | 솔버 인프라 + 데이터 생성 (19K SFT / 20K DPO) | ✅ 라운드트립 100% |
| 4 | Director 리뷰 → context/researcher3 | ✅ |
| 5 | Baseline A1/B1/C1 (12셀) | ✅ [`results/baseline_summary.md`](results/baseline_summary.md) |
| 6 | SFT go/no-go | ✅ GO |
| 7 | Qwen2.5-1.5B SFT (1188 step, loss 0.21, 89분) | ✅ |
| 8 | A2/A3 평가 | ✅ |
| 9 | **최종 분석 — 가설 기각** | ⚠️ [`results/analysis.md`](results/analysis.md) |

### R1 실패 원인

1. **Tool call 호출률 10~16%** — SFT는 한국어 중심, LegalBench는 영어 → 분류 일반화 실패.
2. **Schema 밖 메서드 환각** — 모델이 `equation_check`, `all_values_tac` 등 존재하지 않는 메서드 생성. 제약 디코딩 미적용이 원인.
3. **Final answer 평균 11.5자** — Yes/No 파싱용 접두사(`<lane>`)가 parser를 방해하고, 생성형 태스크는 짧아서 ROUGE 하락.

### R2 (진행 중)

| Step | 내용 | 상태 |
|------|------|------|
| R2-1 | [`context/researcher2_r2.md`](context/researcher2_r2.md) 작성 | ✅ |
| R2-2 | Lane 10→5 축소, 영어 70%, Outlines 통합, final answer ≥80자 | ✅ (실제 평균 639자) |
| R2-3 | SFT 재학습 | 🔄 진행 중 |
| R2-4 | A2/A3 재평가 + R1 대비 개선 측정 | ⏳ |

---

## 결과 요약

### Baseline (R1, balanced-accuracy / ROUGE-L)

| Task | A1 Qwen2.5-1.5B | B1 Qwen3-1.7B | C1 Qwen3 thinking |
|------|-----------------|----------------|--------------------|
| hearsay | **0.522** | 0.509 | 0.431 |
| personal_jurisdiction | 0.591 | 0.500 | **0.663** |
| rule_qa (ROUGE-L) | 0.121 | **0.139** | 0.113 |
| textualism_tool | **0.583** | 0.500 | 0.471 |
| **평균** | **0.454** | 0.412 | 0.419 |

**관찰:**
- Qwen2.5-1.5B가 Qwen3-1.7B CoT를 평균 +4pp 상회 → 1.5B를 주력으로 선택한 것이 타당.
- thinking 모드는 PJ에서만 +16pp로 크게 도움, 다른 태스크에서는 오히려 하락.
- 이 두 관찰 모두 "더 많이 생각"이 만능은 아니라는 기존 가정과 부합.

### A2/A3 (R1, 가설 기각)

| Task | A1 | A2 (+solver) | A3 (+SFT Lane) | Δ (A3−A1) |
|------|----|--------------|-----------------|-----------|
| hearsay | 0.522 | **0.543** | 0.526 | +0.004 |
| PJ | **0.591** | 0.500 | 0.500 | −0.091 |
| rule_qa | 0.121 | **0.176** | 0.092 | −0.029 |
| textualism | **0.583** | 0.444 | 0.489 | −0.094 |
| **평균** | **0.454** | 0.416 | 0.402 | **−0.052** |

- **A3 평균이 A1보다 5pp 낮음** → 핵심 가설 기각.
- 단, **A2가 A1을 이기는 태스크는 존재** (hearsay +2pp, rule_qa +5.5pp) → 솔버 자체 효과는 태스크 선별 시 유효.

### R2 목표

- A3 평균이 최소 B1(0.412)을 넘고, 이상적으로 baseline A1(0.454)도 넘는 것.
- 개별 태스크 단위로 A3 > A2 > A1 구도 재현.

---

## 디렉토리 구조

```
llm-research/
├── README.md                   (본 문서)
├── CLAUDE.md                   Director 지시서 (Claude Code 자동 로드)
│
├── docs/
│   ├── plan.md                 연구 계획 v4 final
│   └── CONTRIBUTING.md         커밋 규약 (Conventional Commits 경량)
│
├── context/                    Director → 연구원 전달 문서
│   ├── researcher2.md          R1 Lane 재배치 지시
│   ├── researcher2_r2.md       R2 Lane 축소 + Outlines
│   └── researcher3.md          R1 평가 설계
│
├── solver/                     Z3 + SymPy 래퍼
│   ├── schemas.py / schemas_r2.py      Lane별 tool_call JSON Schema
│   ├── z3_legal.py             L01~L04 Z3 구현
│   ├── sympy_calc.py           L05 계산
│   ├── executor.py / executor_r2.py    tool_call dispatch
│   ├── validator.py / validator_r2.py  라운드트립 검증
│   └── tests/                  pytest (R1 11/11, R2 21/21 pass)
│
├── data/
│   ├── seeds/ / seeds_r2/      Lane별 시드
│   ├── sft_merged.jsonl        R1 SFT 19K
│   ├── sft_r2.jsonl            R2 SFT 9K (영어 77.5%)
│   ├── dpo_pairs.jsonl         R1 DPO 20K
│   ├── dpo_r2.jsonl            R2 DPO 10K
│   └── legalbench/             LegalBench 4태스크 CSV
│
├── scripts/
│   ├── build_seeds.py          시드 생성
│   ├── generate_data(_r2).py   증강 (시드 × 템플릿)
│   ├── validate_data(_r2).py   품질 검수
│   ├── fetch_datasets.py       HuggingFace 다운로드
│   ├── train_sft.py            transformers + PEFT LoRA
│   ├── train_dpo.py            TRL DPO
│   ├── eval_baseline.py        A1/B1/C1
│   └── eval_with_solver(_r2).py  A2/A3/B2/B3 (--adapter, Outlines)
│
├── configs/                    sft/dpo × qwen25/qwen3 (+r2) yaml
├── models/                     LoRA 어댑터 (git 제외, 2.7GB)
│
├── reports/
│   ├── literature_review.md
│   ├── data_quality.md
│   └── data_quality_r2.md
│
├── results/
│   ├── baseline.csv / baseline_summary.md   A1/B1/C1 12셀
│   ├── experiment_matrix.csv                A2/A3 결과
│   └── analysis.md                          R1 최종 분석
│
└── .claude/agents/             researcher1/2/3 subagent 정의
```

---

## 실행 방법

### 환경 설정

```bash
# 가상환경 (uv 권장)
uv venv .venv
source .venv/bin/activate

uv pip install transformers accelerate bitsandbytes peft trl datasets \
               z3-solver sympy jsonschema pyyaml pandas scikit-learn \
               rouge-score outlines
```

### 솔버 테스트

```bash
python -m pytest solver/tests/       # R1: 11 tests, R2: 21 tests
```

### 데이터 재생성 (필요 시)

```bash
python scripts/build_seeds.py        # Lane별 시드
python scripts/generate_data.py      # 증강 → sft_merged.jsonl, dpo_pairs.jsonl
python scripts/validate_data.py      # 품질 리포트
```

### Baseline 평가 (A1/B1/C1)

```bash
python scripts/eval_baseline.py --model qwen25          --task all
python scripts/eval_baseline.py --model qwen3           --task all
python scripts/eval_baseline.py --model qwen3-thinking  --task all
```

### SFT + DPO

```bash
python scripts/train_sft.py --config configs/sft_qwen25_r2.yaml
python scripts/train_dpo.py --config configs/dpo_qwen25.yaml
```

### 실험군 평가 (A2/A3)

```bash
# A2: 베이스 모델 + solver (Lane 없이)
python scripts/eval_with_solver_r2.py --model qwen25 --variant solver --task all

# A3: SFT 어댑터 + Lane + solver
python scripts/eval_with_solver_r2.py --model qwen25 --variant lane_solver --task all \
       --adapter models/qwen25-1.5b-legal-sft-r2
```

---

## 팀 구성 (Claude Code multi-agent)

이 연구는 Anthropic의 Claude Code CLI를 통해 **Director + 3 subagent** 체계로 수행되고 있습니다.

| 역할 | 모델 | 담당 |
|------|------|------|
| **Director** | Opus 4.7 (opusplan) | 연구 총괄, 리뷰 게이트, context 번역, 최종 분석 |
| **연구원 1** | Sonnet 4.6 | 선행연구 조사 ([정의](.claude/agents/researcher1.md)) |
| **연구원 2** | Sonnet 4.6 | 솔버 인프라 + 데이터 생성/검수 ([정의](.claude/agents/researcher2.md)) |
| **연구원 3** | Haiku 4.5 | 학습·평가 스크립트 작성, 실제 학습 실행 ([정의](.claude/agents/researcher3.md)) |

각 연구원은 `Agent` tool로 일회성 spawn되고, 결과를 파일로 남긴 뒤 종료합니다. Director가 파일을 읽고 다음 단계를 결정합니다.

---

## 알려진 제약 및 이슈

- **SFT 학습 시간**: 3060 12GB + transformers+PEFT (Unsloth 미사용)에서 1 epoch 60~90분.
- **thinking 모드 평가 시간**: Qwen3-thinking이 CoT 대비 ~5배 느림 (max_new_tokens 8배 증가).
- **DPO rejected 설계**: R1에서는 전부 roundtrip fail 케이스라 학습 신호 단순. R2에서 bad_method / bad_matching / short_answer 세 종류로 분산.
- **KLAC (한국어 법률 상담)**: 라이선스 명시 부재로 R1/R2 미사용. 연구 용도로는 활용 가능하나 이번 사이클은 보류.

---

## 데이터 / 선행연구 / 라이선스

### 데이터 출처

- **LegalBench** — `nguha/legalbench` (HuggingFace, CC-BY-4.0). 162 법률 태스크 중 hearsay / personal_jurisdiction / rule_qa / textualism_tool_dictionaries 4개 사용.
- **한국어 법률 상담 QA** — `jihye-moon/klac_legal_aid_counseling` (대한법률구조공단 크롤링). 현재 미사용.
- **합성 데이터** — subagent가 Python 스크립트로 직접 생성 (외부 API 호출 없음).

### 참고 선행연구

- **SatLM** (NeurIPS 2023) — NL → SMT-LIB → Z3. LSAT/GSM에서 +23% (vs PAL).
- **Logic-LM** (EMNLP Findings 2023) — self-refine 포함, 평균 +18.4% vs CoT.
- **NL2Logic** (2026) — AST 경유 NL → FOL → Z3/SMT-LIB. 0.5B에서도 작동.
- **SaulLM** (NeurIPS 2024) — 540B 법률 토큰 pretrain (54B/141B). 파라미터-효율 비교군.
- **LegalBench** (arXiv 2308.11462) — 162 법률 태스크 벤치마크.

### 라이선스

- 코드/문서: MIT 기준(명시 시).
- 연구 용도 (학술 비상업) 기본. 외부 데이터셋은 각 출처 라이선스 준수.

---

## 부록

### A. Git init 이전 편집 이력

이 저장소는 `fb38ebc`(2026-04-19) 커밋으로 초기화되었고, 그 이전의 편집 작업은 Claude Code의 로컬 `~/.claude/file-history/`에 스냅샷으로만 남아 있어 git 이력으로 재구성할 수 없습니다. 재현성 참고용으로 세션별 타임라인:

| 시기 | 세션/내용 | 비고 |
|------|-----------|------|
| 2026-04-08 ~ 04-09 | 초기 탐색 (bonsai-lane-solver) | 1-bit LLM + Lane + 솔버 초안 |
| 2026-04-17 | plan.md v2 → v4 반복 개정 | 범용 Lane → 법률 도메인 pivot |
| 2026-04-18 06:53 ~ 13:36 | plan.md v4 final 확정, 팀 구조 설정 | 608 이벤트 세션 |
| 2026-04-18 13:03 ~ | CLAUDE.md, researcher1/2/3.md 작성 | `.claude/agents/` 구성 |
| 2026-04-19 04:20 ~ | **현재 세션 시작** — Step 1~8 자동 실행 | 550+ 이벤트 |
| 2026-04-19 06:11 | **`fb38ebc` 초기 커밋** | 이 시점부터 git 관리 |

file-history 스냅샷은 Edit/Write로 수정된 파일만 포함하며, subagent가 Python 스크립트로 생성한 대량 산출물(19K+20K 샘플)은 추적되지 않음. 따라서 git 이전 이력을 commit으로 합성하는 것은 **재현성을 오히려 해치므로 지양**했습니다.

### B. `.claude_old/` — 시행착오의 흔적

저장소 루트의 `.claude_old/`는 **폐기된 초기 에이전트 팀 설정**입니다. 현재 `.claude/agents/` 구조로 오기 전의 실험 흔적으로, 학습용으로만 보존되어 있고 **실행되지 않습니다**.

**무엇이었나 (bonsai-lane-solver):**
- 팀 기반 multi-agent (실험적 Agent Teams 기능)
- 구성: `team-lead` (Sonnet) + researcher-lit / data / model (Haiku)
- 통신: SendMessage 기반 pull-first, 연구원별 inbox 파일
- 주제: 당시 plan.md v2.0 — "Bonsai 1-bit LLM + Lane + 솔버" (법률 pivot 이전)

**왜 폐기했나:**
- **30분 만에 토큰 소진**. SendMessage 기반 다자간 통신의 왕복 오버헤드가 실제 연구 진전 대비 과도.
- team-lead가 director 역할까지 겸임해 의례적 상태 체크가 빈발.
- Haiku 연구원의 품질 자가 점검 → "Sonnet 승격 요청" → 비용 이중 발생.

**현재 구조와의 차이:**

| 항목 | `.claude_old/` (폐기) | `.claude/agents/` (현재) |
|------|----------------------|--------------------------|
| 통신 | SendMessage (pull inbox) | Agent tool 단발 호출, 결과 파일로 전달 |
| 오버헤드 | team-lead 상주 + 연구원 상주 | Director는 세션 내 상주, 연구원은 필요 시 spawn |
| 비용 | 토큰 급격 소진 | 한 세션에서 R1+R2 완주 |
| 상태 관리 | inbox JSON 파일 | TaskCreate/TaskUpdate + 파일 산출물 |

**보존 이유:** 초기 팀 프롬프트에 **당시 설계 의도**가 남아 있어 multi-agent 실험 재시도 시 "무엇이 비쌌는지" 비교 기준으로 활용 가능.
