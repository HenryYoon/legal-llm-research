# 법률 추론 특화 Lane + 솔버 소형 LLM 연구 계획

**2026.04 — 희승 (Henry Yoon)**
**v4 final — 법률 도메인 → 범용 확장 로드맵**

---

## 1. 핵심 가설

> "법률 추론을 Lane 구조로 분류하고 형식화 가능한 추론을 Z3 솔버로 외주하면, 1.5~1.7B 모델이 법률 벤치마크에서 7B+ CoT 모델을 이길 수 있다."

법률 도메인을 먼저 선택한 이유:
- 법률 추론은 **연역 추론의 비율이 높아** 솔버 효과가 극대화됨
- 요건 매칭, 포섭, 규칙 적용이 **형식 논리로 직접 변환 가능**
- LegalBench (`nguha/legalbench`)와 한국어 법률 상담 QA (`jihye-moon/klac_legal_aid_counseling`) 데이터를 활용
- 법률은 lane 간 경계가 명확해서 분류 정확도가 높을 것으로 예상

---

## 2. 베이스 모델

**Qwen2.5-1.5B-Instruct** (주력) — SFT 파이프라인 검증됨, Unsloth 호환
**Qwen3-1.7B** (비교군) — thinking 모드 내장, baseline 높음

---

## 3. 아키텍처

```
[Qwen 1.5~1.7B — 법률 Lane SFT 적용]
├── Legal Lane 분류 ← SFT
├── Tool call 생성 (Z3 솔버 호출) ← SFT
└── 직접 생성 (법률 설명, 요약) ← 기존 능력 + SFT

[외부 모듈]
├── Z3 (요건 매칭, 포섭, 논리 추론)
├── SymPy (손해배상 계산 등 수치 추론)
└── 제약 디코딩: Outlines/XGrammar
```

---

## 4. 법률 Lane 분류 체계

### 4.1 형식화 가능 Lane (솔버 경유)

| Lane | 설명 | Tool | 출력 스키마 | LegalBench 대응 |
|------|------|------|-----------|----------------|
| L01 요건 매칭 | 법적 요건(element)과 사실관계 매칭 | Z3 | `{elements, facts, matching, all_met}` | hearsay, personal_jurisdiction |
| L02 규칙 적용 | 조문/규칙을 사실에 적용 | Z3 | `{rule, conditions, facts, conclusion}` | rule_qa, statutory_reasoning |
| L03 포섭 | 사실을 법적 규범에 포섭 | Z3 | `{norm, facts, subsumption, holding}` | textualism_tool |
| L04 논리 판단 | 법적 논증의 타당성 판단 | Z3 | `{premises, inference, valid}` | — |
| L05 계산 | 손해배상, 이자, 기간 계산 | SymPy | `{expr, result}` | — |

### 4.2 직접 생성 Lane (모델)

| Lane | 설명 |
|------|------|
| L06 법률 설명 | 조문/판례의 의미 설명 |
| L07 사례 비교 | 유사 사례 간 차이점 분석 |
| L08 요약 | 판결문/계약서 요약 |
| L09 번역 | 법률 용어 한↔영 변환 |
| L10 불확실 | 판단 불가 시 "추가 정보 필요" |

### 4.3 왜 법률이 Lane+솔버에 최적인가

법률 추론의 본질적 구조:
```
사실관계 확인 → 적용 법률 특정 → 요건 매칭 → 포섭 → 결론
```
이것이 정확히:
```
입력 파싱 → Lane 분류(L01~L05) → tool_call 생성 → Z3 실행 → 자연어 결론
```
과 1:1 대응됨. 범용 도메인과 달리 **"이 입력이 형식화 가능한가?"의 판단이 쉬움** — 법률 질문은 거의 전부 형식화 가능.

---

## 5. 실험 매트릭스

|  | CoT만 | +솔버 | +Lane+솔버 |
|--|-------|------|-----------|
| Qwen2.5-1.5B | A1 | A2 | A3 |
| Qwen3-1.7B | B1 | B2 | B3 |
| Qwen3-1.7B thinking | C1 | — | — |

검증 가설:
- A2 > A1 → 법률 추론에서 솔버 효과 존재
- A3 > A2 → Legal Lane이 효과 증폭
- A3 > B1 → 1.5B+Lane+솔버 > 1.7B CoT (법률 벤치마크)
- A3 vs C1 → 외부 솔버 vs 내부 thinking (법률 추론)

---

## 6. 데이터

### 6.1 기존 확보 데이터

| 데이터셋 | 규모 | 용도 |
|---------|------|------|
| LegalBench (`nguha/legalbench`) | 다수 태스크 | 평가 + SFT 소스 |
| — hearsay | ~수백 | L01 요건 매칭 |
| — rule_qa | ~수백 | L02 규칙 적용 |
| — personal_jurisdiction | ~수백 | L01 요건 매칭 |
| — textualism_tool | ~수백 | L03 포섭 |
| 한국어 법률 상담 QA (`jihye-moon/klac_legal_aid_counseling`) | HuggingFace | L06 설명 + 다국어 확장 |

### 6.2 합성 데이터 (subagent가 생성)

별도 API 호출 없음. 연구원 2(sonnet subagent)가 직접:

1. **시드 NL 예시 생성:** Lane당 50개 × 10 lane = 500개 (~15K 토큰)
2. **솔버 입력 규칙 기반 랜덤 생성:** Python 스크립트로 법률 요건/사실 조합 자동 생성
3. **템플릿 변형 스크립트:** 시드를 수치/이름/상황 변형으로 증강
4. **라운드트립 검증:** NL → tool_call → Z3 → 결과 일치 확인

목표 규모:

| 유형 | 샘플 수 |
|------|--------|
| SFT: Lane 분류 | 10 lane × 2K = 20K |
| SFT: Tool call (L01~L05) | 30K |
| SFT: 직접 생성 (L06~L10) | 10K |
| DPO: (chosen, rejected) 쌍 | 20K |
| **총합** | **~80K** |

### 6.3 샘플 형식

```json
{
  "messages": [
    {"role": "system", "content": "너는 법률 추론 AI야. 입력을 Legal Lane으로 분류하고 적절한 솔버를 호출해라."},
    {"role": "user", "content": "피고인이 편의점에서 음료수 2캔을 주머니에 넣고 나왔다. 절도죄가 성립하는가?"},
    {"role": "assistant", "content": "<lane>L01_element_matching</lane>\n<tool_call>{\"tool\": \"z3\", \"method\": \"check_elements\", \"elements\": [\"타인의 재물\", \"절취\"], \"facts\": [\"편의점 소유 음료수\", \"주머니에 넣고 퇴장→점유 이전\"]}</tool_call>"},
    {"role": "tool", "content": "{\"all_met\": true, \"details\": {\"타인의 재물\": true, \"절취\": true}}"},
    {"role": "assistant", "content": "절도죄(형법 제329조)의 구성요건이 모두 충족되어 성립합니다. 편의점 음료수는 '타인의 재물'에 해당하고, 주머니에 넣고 퇴장한 행위는 '절취'에 해당합니다."}
  ]
}
```

---

## 7. 가상 연구실 구성

```
┌──────────────────────────────────────────────────────┐
│            연구 총괄 Director (opusplan)               │
│  방향 결정 / 리뷰 게이트 / 맥락 번역 / 최종 분석       │
└──────────┬───────────────┬───────────────┬────────────┘
           │               │               │
     ┌─────▼──────┐  ┌─────▼──────┐  ┌─────▼──────┐
     │  연구원 1   │  │  연구원 2   │  │  연구원 3   │
     │  선행연구   │  │   데이터    │  │   모델링    │
     │  (sonnet)  │  │  (sonnet)  │  │  (haiku)   │
     └────────────┘  └────────────┘  └────────────┘
```

### 7.1 Director — 연구 총괄

**모델:** `opusplan` (판단=opus 자동, 실행=sonnet 자동)

**역할:**
- 연구 질문 정의 및 실험 설계
- 각 연구원 산출물 리뷰 및 go/no-go 판단
- **맥락 번역:** 연구원 1의 조사 결과에서 연구원 2/3에 필요한 부분을 추려 `context_for_*.md`로 전달
- 최종 결과 해석 및 결론 도출

**프롬프트:**
```
너는 법률 AI 연구실의 총괄 책임자(PI)야.
3명의 연구원을 지휘하여 "Legal Lane + Z3 솔버로 소형 LLM 법률 추론 강화" 연구를 수행한다.

실행 순서:
  Step 1 → 연구원 1: 선행연구 조사
  Step 2 → 리뷰 → context_for_researcher2.md 작성
  Step 3 → 연구원 2: 솔버 인프라 + 데이터 생성
  Step 4 → 리뷰 → context_for_researcher3.md 작성
  Step 5 → 연구원 3: baseline 측정 (A1, B1, C1)
  Step 6 → 리뷰 → 학습 진행 판단
  Step 7 → 연구원 3: SFT + DPO 학습
  Step 8 → 연구원 3: 실험군 평가 (A2, A3, B2, B3)
  Step 9 → 최종 분석 → results/analysis.md

판단 기준:
- 선행연구에서 동일 실험 존재 → 차별화 포인트 재설정
- 데이터 라운드트립 통과율 < 85% → 연구원 2에게 수정 지시
- A2 ≤ A1 → 프롬프트 수정 or 솔버 스키마 단순화
- A3 ≤ B1 → Lane 수 축소 (10→5), 스키마 재설계

맥락 번역 예시:
  연구원 1이 "LegalBench의 hearsay 태스크는 binary classification이라 Z3보다
  직접 분류가 나을 수 있다"고 보고하면 → context_for_researcher2.md에
  "hearsay는 L01이 아니라 L06(직접 생성)으로 재분류하라" 지시
```

### 7.2 연구원 1 — 선행연구 조사/검증

**모델:** `sonnet` (subagent 기본)

**프롬프트:**
```
너는 법률 AI 연구실의 선행연구 담당 연구원이야.

태스크 — 초기 조사:
다음을 웹 검색으로 조사하고 reports/literature_review.md에 정리해라:

a. 법률 추론에 형식 솔버(Z3/SAT/SMT)를 적용한 기존 연구
   - SatLM, NL2Logic 등에서 법률 태스크 결과가 있는가
   - 소형 모델(1~3B)에서의 실험이 있는가
b. LegalBench 태스크별 특성
   - hearsay, rule_qa, personal_jurisdiction, textualism_tool
   - 각 태스크가 형식화(Z3) 가능한지, 직접 분류가 나은지 판단
c. Qwen2.5-1.5B / Qwen3-1.7B의 법률 관련 벤치마크 (있으면)
d. 법률 도메인 SFT 관련 최신 연구
   - SaulLM, LegalBERT 등 법률 특화 모델의 접근법
e. LegalBench 다운로드 및 평가 코드 위치

각 항목: 논문명, 연도, 핵심 수치, 이 연구에 대한 시사점 1~2줄, 출처 URL.
불확실한 건 "미확인"으로 표기하고 추가 검색 하지 마라.
```

### 7.3 연구원 2 — 데이터 생성/검수

**모델:** `sonnet` (subagent 기본)

**프롬프트:**
```
너는 법률 AI 연구실의 데이터 담당 연구원이야.
Director가 전달한 context_for_researcher2.md를 반드시 먼저 읽어라.

태스크 1 — 솔버 인프라:
solver/ 디렉토리에 법률 추론용 Z3 래퍼를 구현해라:
  solver/__init__.py
  solver/z3_legal.py     — 요건 매칭(L01), 규칙 적용(L02), 포섭(L03), 논리 판단(L04)
  solver/sympy_calc.py   — 법률 계산(L05): 손해배상, 이자, 기간
  solver/schemas.py      — Legal Lane별 tool call JSON Schema
  solver/executor.py     — tool_call JSON → 솔버 실행 → 결과
  solver/validator.py    — 라운드트립 검증

태스크 2 — 시드 데이터 직접 생성:
Legal Lane별로 시드 NL 예시를 직접 작성해라 (API 호출 하지 마라):
  L01 요건 매칭: 50개 (절도, 사기, 폭행 등 구성요건 매칭)
  L02 규칙 적용: 50개 (조문 조건→결론 적용)
  L03 포섭: 50개 (사실→규범 포섭)
  L04 논리 판단: 50개 (법적 논증 타당성)
  L05 계산: 50개 (손해배상, 이자 계산)
  L06~L10 직접 생성: 각 30개
저장: data/seeds/ 디렉토리에 lane별 JSON

태스크 3 — 증강 스크립트:
scripts/generate_data.py를 작성해라:
  1. 시드 × 템플릿 변형 (수치, 이름, 상황 교체) → 대량 증강
  2. 솔버 입력 랜덤 생성 → 정답 산출 → 시드 패턴으로 NL 조합
  3. 라운드트립 검증
  4. JSONL 출력 (SFT + DPO)

태스크 4 — 품질 검수:
scripts/validate_data.py → reports/data_quality.md:
  라운드트립 통과율, Lane별 분포, 스키마 준수율

출력: solver/, data/, scripts/generate_data.py, scripts/validate_data.py, reports/data_quality.md
```

### 7.4 연구원 3 — 모델링 (학습/평가)

**모델:** `haiku` (frontmatter override)

**프롬프트:**
```
너는 법률 AI 연구실의 모델링 담당 연구원이야.
Director가 전달한 context_for_researcher3.md를 반드시 먼저 읽어라.

태스크 1 — 학습 스크립트:
  scripts/train_sft.py  (--model qwen25 | qwen3)
  scripts/train_dpo.py  (--model qwen25 | qwen3)
  config는 configs/ 디렉토리의 yaml을 읽도록 구현

태스크 2 — 평가 스크립트:
  scripts/eval_baseline.py     — A1, B1, C1
  scripts/eval_with_solver.py  — A2, A3, B2, B3

  평가 대상:
    LegalBench 태스크 (hearsay, rule_qa, personal_jurisdiction, textualism_tool)
    한국어 법률 상담 QA (jihye-moon/klac_legal_aid_counseling, HuggingFace)

  솔버 연동 평가 로직:
    1. 모델에 tool_call 생성 프롬프트
    2. <tool_call> 파싱 → solver/executor.py 실행
    3. 최종 답 → 정답 비교
    4. 파싱 실패 시 fallback: 모델 직접 답변

태스크 3 — 결과 집계:
  results/experiment_matrix.csv
  results/comparison_table.md
  results/analysis.md
```

---

## 8. 실행 흐름

```
Step 1: Director → 연구원 1
  "법률 도메인 선행연구 조사해라"
  → reports/literature_review.md

Step 2: Director(opus) 리뷰 ★
  - LegalBench 태스크 중 Z3에 적합한 것/부적합한 것 판별
  - 기존 법률+솔버 연구와의 차별점 정리
  → context_for_researcher2.md

Step 3: Director → 연구원 2
  "솔버 인프라 + 법률 데이터 생성해라" + context 첨부
  → solver/, data/, reports/data_quality.md

Step 4: Director(opus) 리뷰 ★
  - 데이터 품질 확인 (통과율 85%+)
  - Lane별 분포 균형 확인
  → context_for_researcher3.md

Step 5: Director → 연구원 3
  "baseline 측정해라 (A1, B1, C1)" + context 첨부
  → results/ (baseline)

Step 6: Director 확인 → baseline이 예상 범위 내인가

Step 7: Director → 연구원 3
  "SFT + DPO 학습해라" ⚠️ GPU 전용

Step 8: Director → 연구원 3
  "실험군 평가해라 (A2, A3, B2, B3)"
  → results/ (전체)

Step 9: Director(opus) 최종 분석 ★
  - 가설 검증
  - 예상 밖 결과 시 → 연구원 1에게 추가 조사 요청 가능
  → results/analysis.md
```

---

## 9. 프로젝트 설정

### 9.1 프로젝트 초기화 시 생성할 파일들

프로젝트 루트에서 다음을 생성:

```
lane-solver-legal/
├── .claude/
│   ├── settings.json          # Claude Code 프로젝트 설정
│   └── agents/
│       ├── researcher1.md     # 연구원 1 frontmatter + 프롬프트
│       ├── researcher2.md     # 연구원 2 frontmatter + 프롬프트
│       └── researcher3.md     # 연구원 3 frontmatter + 프롬프트
├── CLAUDE.md                  # Director 지시서
├── configs/
│   ├── sft_qwen25.yaml
│   ├── sft_qwen3.yaml
│   ├── dpo_qwen25.yaml
│   └── dpo_qwen3.yaml
├── solver/
├── data/
│   └── seeds/
├── scripts/
├── models/
├── results/
└── reports/
```

### 9.2 .claude/settings.json

```json
{
  "model": "opusplan",
  "env": {
    "MAX_THINKING_TOKENS": "10000",
    "CLAUDE_CODE_SUBAGENT_MODEL": "sonnet",
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

- `opusplan`: Director가 판단은 opus, 실행은 sonnet 자동 전환
- `CLAUDE_CODE_SUBAGENT_MODEL`: 연구원 1, 2의 기본 모델 (sonnet)
- `MAX_THINKING_TOKENS`: 10K (기본 ~32K 대비 ~70% 절감)
- 연구원 3만 frontmatter로 haiku override

### 9.3 .claude/agents/researcher1.md

```markdown
---
effort: medium
---

# 연구원 1: 선행연구 조사

## 역할
법률 도메인 + 솔버 + 소형 LLM 관련 선행연구를 조사하고 정리한다.

## 출력
reports/literature_review.md에 마크다운으로 정리.
항목당 3줄 이내, 팩트 중심, 불확실한 건 "미확인".

## 조사 항목
- 법률 추론 + 형식 솔버(Z3/SAT) 기존 연구
- LegalBench 태스크별 특성 및 Z3 적합성
- Qwen2.5-1.5B / Qwen3-1.7B 법률 벤치마크
- 법률 도메인 SFT 최신 연구 (SaulLM, LegalBERT 등)
- LegalBench 다운로드 및 평가 코드 위치
```

### 9.4 .claude/agents/researcher2.md

```markdown
---
effort: high
---

# 연구원 2: 데이터 생성/검수

## 역할
Z3 솔버 인프라 구축 + 법률 학습 데이터 생성 + 품질 검수.
context_for_researcher2.md를 먼저 읽고 시작할 것.

## 핵심 원칙
- 별도 API 호출 금지. 시드 NL은 직접 생성.
- 대량 증강은 Python 스크립트로 로컬 실행.
- 모든 tool_call 샘플은 라운드트립 검증 통과 필수.

## 출력
solver/, data/, scripts/generate_data.py, scripts/validate_data.py, reports/data_quality.md
```

### 9.5 .claude/agents/researcher3.md

```markdown
---
effort: low
model: haiku
---

# 연구원 3: 모델링

## 역할
학습 스크립트 + 평가 스크립트 작성.
context_for_researcher3.md를 먼저 읽고 시작할 것.

## 핵심 원칙
- 스크립트만 생성, 실행은 bash로.
- configs/ 디렉토리의 yaml을 읽는 구조로 구현.
- OOM 시 자동 fallback: batch 절반 + grad_accum 2배.

## 출력
scripts/train_sft.py, scripts/train_dpo.py, scripts/eval_baseline.py, scripts/eval_with_solver.py
```

### 9.6 CLAUDE.md (Director 지시서)

```markdown
# Legal Lane + Solver Research

## 연구 목표
법률 추론에서 1.5B 모델 + Lane + Z3 솔버가 7B CoT를 이기는 것을 증명

## 팀
- Director (opusplan): 연구 총괄
- 연구원 1 (sonnet): 선행연구 — .claude/agents/researcher1.md
- 연구원 2 (sonnet): 데이터 — .claude/agents/researcher2.md
- 연구원 3 (haiku): 모델링 — .claude/agents/researcher3.md

## 환경
- GPU: RTX 3060 12GB
- Python 3.10+
- pip install unsloth trl z3-solver sympy outlines datasets

## 실행 순서
1. 연구원 1 → 선행연구 조사
2. Director 리뷰 → context_for_researcher2.md
3. 연구원 2 → 솔버 + 데이터
4. Director 리뷰 → context_for_researcher3.md
5. 연구원 3 → baseline (A1, B1, C1)
6. Director 리뷰
7. 연구원 3 → SFT + DPO (GPU 전용)
8. 연구원 3 → 실험군 (A2, A3, B2, B3)
9. Director → 최종 분석

## VRAM 제약
- 학습 시 GPU 독점
- 추론 4bit 양자화 필수
- batch 4, OOM 시 2로 축소

## 데이터
- LegalBench: nguha/legalbench (HuggingFace)
- 한국어 법률 상담 QA: jihye-moon/klac_legal_aid_counseling (HuggingFace)
- 합성 데이터: subagent 직접 생성, 별도 API 호출 없음

## 판단 기준
- 데이터 라운드트립 통과율 < 85% → 수정
- A2 ≤ A1 → 솔버 스키마 단순화
- A3 ≤ B1 → Lane 축소 (10→5)
```

### 9.7 configs/sft_qwen25.yaml

```yaml
model_name: unsloth/Qwen2.5-1.5B-Instruct
output_dir: models/qwen25-1.5b-legal-sft
load_in_4bit: true
lora_r: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-4
max_seq_length: 2048
bf16: true
gradient_checkpointing: true
dataset_path: data/sft_merged.jsonl
```

### 9.8 configs/dpo_qwen25.yaml

```yaml
model_name: models/qwen25-1.5b-legal-sft
output_dir: models/qwen25-1.5b-legal-dpo
beta: 0.1
num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5e-5
max_length: 2048
dataset_path: data/dpo_pairs.jsonl
```

(qwen3 버전은 model_name과 output_dir만 변경)

---

## 10. 벤치마크

| 벤치마크 | 측정 대상 | 우선순위 |
|---------|---------|---------|
| LegalBench (hearsay) | 요건 매칭 | 최우선 |
| LegalBench (rule_qa) | 규칙 적용 | 최우선 |
| LegalBench (personal_jurisdiction) | 요건 매칭 | 높음 |
| LegalBench (textualism_tool) | 포섭 | 높음 |
| 한국어 법률 상담 QA (klac) | 다국어 법률 추론 | 중간 |

---

## 11. 예상 결과

| 태스크 | Qwen2.5-1.5B 단독 | +Lane+솔버 | 향상 근거 |
|-------|------------------|-----------|---------|
| hearsay | ~55~65% | **80~90%** | 요건 매칭 → Z3 직접 판정 |
| rule_qa | ~50~60% | **85~95%** | 규칙 조건→결론 → Z3 직접 판정 |
| personal_jurisdiction | ~50~60% | **75~85%** | 다요건 매칭 → Z3 |
| textualism_tool | ~45~55% | **70~80%** | 포섭 → Z3 |

법률 도메인은 형식화 비율이 높아서 솔버 효과가 범용 벤치마크(GSM8K 등)보다 **더 극적**일 것으로 예상.

---

## 12. 체크포인트

| 체크 | 확인 사항 | 실패 시 |
|------|---------|--------|
| 선행연구 | 동일 실험 존재 여부 | 차별화 재설정 |
| solver/ 테스트 | Z3 법률 래퍼 정상 동작 | 스키마 단순화 |
| 데이터 품질 | 라운드트립 통과율 85%+ | 파이프라인 수정 |
| A2 > A1 | 솔버 경유 > CoT (법률) | 솔버 스키마 단순화 |
| A3 > B1 | 1.5B+Lane+솔버 > 1.7B CoT | Lane 5개로 축소 |

---

## 13. 선행 연구

**법률 + 솔버:** SatLM (NeurIPS 2023), NL2Logic (2025)
**법률 특화 모델:** SaulLM, LegalBERT, LawGPT
**법률 벤치마크:** LegalBench (nguha/legalbench)
**솔버 일반:** PAL (ICML 2023), PoT (TMLR 2023)
**Lane/라우팅:** DinoDS, Semantic Router, MoDEM (2024)
**소형 Tool Calling:** TinyAgent (Berkeley 2024), BFCL V4 (ICML 2025)
**제약 디코딩:** DCCD (2025), Outlines, XGrammar

---

## 14. 장기 확장

법률 도메인에서 효과 확인 후:
1. **범용 확장:** 법률 Lane(10개) → 범용 Lane(20~30개)
2. **다국어:** 한국어 법률 상담 QA(klac)로 다국어 법률 추론
3. **사전학습:** Lane 기반 법률 코퍼스 큐레이션으로 사전학습 효율 검증
4. **1-bit:** Bonsai 모델에 Legal Lane SFT 적용
5. **Coconut:** 잠재 공간 추론 + 솔버 결합
6. **MoE/MoD:** 법률 Lane을 expert로 내재화