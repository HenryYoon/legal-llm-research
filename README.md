# Legal Lane + Solver Research

**법률 추론에서 1.5B 모델 + Lane 분류 + Z3 솔버가 7B CoT 모델을 이기는 것을 증명하는 연구**

> "법률 추론을 Lane 구조로 분류하고 형식화 가능한 추론을 Z3 솔버로 외주하면,
> 1.5~1.7B 모델이 법률 벤치마크에서 7B+ CoT 모델을 이길 수 있다."

---

## 아키텍처

```
[Qwen 1.5~1.7B — Legal Lane SFT/DPO]
├── Legal Lane 분류           ← SFT
├── tool_call 생성 (Z3/SymPy) ← SFT
└── 직접 생성 (설명, 요약)    ← SFT + 기존 능력

[외부 솔버]
├── Z3          요건 매칭, 규칙 적용, 포섭, 논리 판단
└── SymPy       손해배상/이자/기간 계산
```

### Lane 체계

| Lane | 설명 | 경로 |
|------|------|------|
| L01 요건 매칭 | 요건(elements) vs 사실(facts) | Z3 |
| L02 규칙 적용 | 조문 조건 → 결론 | Z3 |
| L03 포섭 | 사실 → 규범 포섭 | Z3 |
| L04 논리 판단 | 논증 타당성 | Z3 |
| L05 계산 | 손해배상/이자 | SymPy |
| L06 설명 | 조문/판례 의미 | 모델 |
| L07 사례 비교 | 유사 사례 차이 | 모델 |
| L08 요약 | 판결문/계약서 | 모델 |
| L09 번역 | 법률 용어 한↔영 | 모델 |
| L10 불확실 | 판단 불가 | 모델 |

---

## 환경

- GPU: RTX 3060 12GB
- Python 3.10+ (실측 3.12)
- 가상환경: `uv venv .venv` (주의: pip 대신 `uv pip install` 사용)

### 의존성

```bash
source .venv/bin/activate
uv pip install transformers accelerate bitsandbytes peft trl datasets \
               z3-solver sympy jsonschema pyyaml pandas scikit-learn \
               rouge-score
```

베이스 모델:
- Qwen/Qwen2.5-1.5B-Instruct (주력)
- Qwen/Qwen3-1.7B (비교군, thinking 모드 포함)

---

## 디렉토리 구조

```
llm-research/
├── CLAUDE.md                       Director 지시서 (Claude Code 루트 자동 로드)
├── README.md                       (본 문서)
├── docs/
│   └── plan.md                     연구 계획 (v4 final)
├── context/                        Director → 연구원 전달 문서
│   ├── researcher2.md              R1 Lane 재배치
│   ├── researcher2_r2.md           R2 Lane 축소 + Outlines
│   └── researcher3.md              R1 평가 설계
├── configs/                        sft/dpo × qwen25/qwen3 yaml
├── solver/                         Z3 + SymPy 법률 추론 래퍼
│   ├── schemas.py                  Lane별 tool_call JSON Schema
│   ├── z3_legal.py                 L01~L04 Z3 구현
│   ├── sympy_calc.py               L05 SymPy 구현
│   ├── executor.py                 tool_call dispatch
│   ├── validator.py                라운드트립 검증
│   └── tests/test_basic.py         pytest 11/11 pass
├── data/
│   ├── seeds/                      Lane별 시드 404개
│   ├── sft_merged.jsonl            19,000 SFT 샘플 (라운드트립 100%)
│   ├── dpo_pairs.jsonl             20,000 DPO 쌍
│   └── legalbench/                 LegalBench 4태스크 CSV
├── scripts/
│   ├── build_seeds.py              시드 생성
│   ├── generate_data.py            증강 (시드 × 템플릿 변형)
│   ├── validate_data.py            품질 검수
│   ├── fetch_datasets.py           HuggingFace 데이터 다운로드
│   ├── train_sft.py                transformers + PEFT LoRA SFT
│   ├── train_dpo.py                TRL DPO
│   ├── eval_baseline.py            A1/B1/C1 평가
│   └── eval_with_solver.py         A2/A3/B2/B3 평가 (+ --adapter)
├── models/                         LoRA 어댑터 (git 제외)
├── reports/
│   ├── literature_review.md        선행연구 조사
│   └── data_quality.md             데이터 품질 리포트
├── results/
│   ├── baseline.csv                A1/B1/C1 12셀
│   ├── baseline_summary.md         baseline 분석
│   └── experiment_matrix.csv       A2/A3 결과 (진행 중)
└── .claude/agents/                 researcher1/2/3 subagent 정의
```

---

## 연구 진행 현황 (2026-04-19 기준)

| Step | 내용 | 상태 |
|------|------|------|
| 1 | 선행연구 조사 (연구원 1) | ✅ `reports/literature_review.md` |
| 2 | Director 리뷰 → context/researcher2 | ✅ Lane 재배치 확정 |
| 3 | 솔버 인프라 + 데이터 생성 (연구원 2) | ✅ 라운드트립 100% |
| 4 | Director 리뷰 → context/researcher3 | ✅ |
| 5 | Baseline A1/B1/C1 측정 (연구원 3) | ✅ 12셀 완료 |
| 6 | Director 리뷰 → SFT go/no-go | ✅ GO |
| 7 | SFT 학습 (Qwen2.5-1.5B) | ✅ loss 0.21, 89분 |
| 7b | DPO 학습 | ⏳ 예정 |
| 8 | 실험군 A2/A3 평가 | 🔄 실행 중 |
| 9 | 최종 분석 | ⏳ |

### Baseline 결과 (balanced-accuracy / ROUGE-L)

| Task | A1 Qwen2.5-1.5B | B1 Qwen3-1.7B | C1 Qwen3 thinking |
|------|-----------------|----------------|--------------------|
| hearsay | **0.522** | 0.509 | 0.431 |
| personal_jurisdiction | 0.591 | 0.500 | **0.663** |
| rule_qa (ROUGE-L) | 0.121 | **0.139** | 0.113 |
| textualism_tool | **0.583** | 0.500 | 0.471 |
| **Mean** | **0.454** | 0.412 | 0.419 |

- Parse rate: 100% 전 셀.
- Qwen2.5-1.5B가 Qwen3-1.7B CoT를 mean 기준 상회 → 주력 모델 선택 타당.
- thinking 모드는 personal_jurisdiction에서만 +16.3pp 유의미.

### Lane 재배치 결정 (plan.md 대비 변경)

선행연구에서 rule_qa/textualism_tool이 형식화에 부적합함을 확인:

| 태스크 | 원래 | 수정 | 근거 |
|--------|------|------|------|
| hearsay | L01 | L01 (유지) | Boolean AND 요건 |
| personal_jurisdiction | L01 | L01 (유지) | 다요건 AND/OR |
| rule_qa | L02 | **L06 (직접 생성)** | rule-recall(암기) |
| textualism_tool | L03 | **L06 (직접 생성)** | 수사 분류 |

---

## 실험 매트릭스

|  | CoT | +Solver | +Lane+Solver |
|--|-----|---------|--------------|
| Qwen2.5-1.5B | A1 ✅ | A2 🔄 | A3 🔄 |
| Qwen3-1.7B | B1 ✅ | B2 ⏳ | B3 ⏳ |
| Qwen3-1.7B thinking | C1 ✅ | — | — |

검증 가설:
- A2 > A1 → 법률 추론에서 솔버 효과 존재
- A3 > A2 → Lane 분류가 솔버 효과 증폭
- **A3 > B1 → 1.5B+Lane+솔버 > 1.7B CoT (핵심)**
- A3 vs C1 → 외부 솔버 vs 내부 thinking

---

## 실행 방법

### 솔버 유닛 테스트
```bash
source .venv/bin/activate
python -m pytest solver/tests/
```

### 데이터 재생성
```bash
python scripts/build_seeds.py
python scripts/generate_data.py
python scripts/validate_data.py
```

### Baseline 평가
```bash
python scripts/eval_baseline.py --model qwen25 --task all
python scripts/eval_baseline.py --model qwen3 --task all
python scripts/eval_baseline.py --model qwen3-thinking --task all
```

### SFT + DPO
```bash
python scripts/train_sft.py --config configs/sft_qwen25.yaml
python scripts/train_dpo.py --config configs/dpo_qwen25.yaml
```

### 실험군 평가
```bash
# A2: 베이스 모델 + 솔버
python scripts/eval_with_solver.py --model qwen25 --variant solver --task all

# A3: SFT 어댑터 + Lane + 솔버
python scripts/eval_with_solver.py --model qwen25 --variant lane_solver --task all \
    --adapter models/qwen25-1.5b-legal-sft
```

---

## 팀 구성 (Claude Code multi-agent)

- **Director (opusplan)**: 연구 총괄, 리뷰 게이트, 맥락 번역
- **연구원 1 (sonnet)**: 선행연구 — `.claude/agents/researcher1.md`
- **연구원 2 (sonnet)**: 데이터/솔버 — `.claude/agents/researcher2.md`
- **연구원 3 (haiku)**: 모델링 — `.claude/agents/researcher3.md`

---

## 알려진 제약 / 이슈

- **SFT 최종 답변 텍스트 부실**: L01~L05 샘플의 tool 실행 이후 `assistant` 메시지 평균 11.5자로 짧음. binary 태스크에는 큰 문제 없으나 생성형 태스크(rule_qa) 및 한국어 상담 SFT 확장 시 재생성 필요.
- **DPO rejected가 전부 roundtrip fail**: 학습 신호는 명확하지만 "포맷 오류 회피"만 학습될 수 있음.
- **KLAC (한국어 법률 상담)**: 연구 목적 사용은 가능하나 이번 사이클은 제외. 다음 라운드에서 L06 강화와 함께 편입 예정.
- **학습 시간**: 3060 12GB + transformers+PEFT (Unsloth 미사용)에서 SFT 1 epoch ~90분.
- **thinking 모드 시간**: Qwen3-thinking 평가가 CoT 대비 ~5배 느림 (max_new_tokens 8배 증가).

---

## 데이터 출처

- **LegalBench**: `nguha/legalbench` (HuggingFace, CC-BY-4.0)
  - hearsay, personal_jurisdiction, rule_qa, textualism_tool_dictionaries
- **한국어 법률 상담 QA**: `jihye-moon/klac_legal_aid_counseling` (대한법률구조공단 크롤링, 현재 미사용)
- **합성 데이터**: subagent 직접 생성, 별도 API 호출 없음

---

## 참고 선행연구

- **SatLM** (NeurIPS 2023) — NL → SMT-LIB → Z3. LSAT에서 효과 검증.
- **Logic-LM** (EMNLP Findings 2023) — self-refine 포함, +18.4% vs CoT.
- **NL2Logic** (2026) — AST 경유 NL→FOL→Z3. 0.5B 모델에서도 작동.
- **SaulLM** (NeurIPS 2024) — 540B 법률 토큰 pretrain (54B/141B). 파라미터-효율 비교군.
- **LegalBench** (arXiv 2308.11462) — 162 법률 태스크 벤치마크.

---

## 라이선스

연구용 (학술 비상업). 외부 데이터셋의 라이선스는 각 출처의 조건 준수.

---

## 부록: Git init 이전 편집 이력

이 저장소는 2026-04-19 `fb38ebc` 커밋으로 초기화되었고, 그 이전의 편집 작업은 Claude Code의 로컬 file-history(`~/.claude/file-history/`)에 스냅샷 형태로만 남아 있어 git 이력으로 재구성할 수 없음. 재현성 참고용으로 세션별 타임라인 정리:

| 시기 | 세션/내용 | 비고 |
|------|-----------|------|
| 2026-04-08 ~ 04-09 | 초기 탐색 (bonsai-lane-solver 주제) | 1-bit LLM + Lane + 솔버 초안 |
| 2026-04-17 | plan.md v2 → v4 반복 개정 | 범용 Lane → 법률 도메인 pivot |
| 2026-04-18 06:53 ~ 13:36 | plan.md v4 final 확정, 팀(subagent) 구조 설정 | 608 이벤트 세션 |
| 2026-04-18 13:03 ~ | CLAUDE.md, researcher1/2/3.md 작성 | `.claude/agents/` 구성 |
| 2026-04-18 14:30 | Step 1 선행연구 세션 준비 | literature review 구도 |
| 2026-04-19 04:20 ~ | **현재 세션 시작** — Step 1~8 자동 실행 | 550+ 이벤트 |
| 2026-04-19 06:11 | **`fb38ebc` 초기 커밋** | 이 시점부터 git 관리 |

file-history 하위 스냅샷(hash@vN)은 Edit/Write로 수정된 파일만 포함하며, subagent가 Python 스크립트로 생성한 대량 산출물(`data/sft_merged.jsonl` 등 19K+20K 샘플)은 추적 대상이 아니었음. 따라서 git 이전 이력을 commit으로 합성하는 것은 재현성을 오히려 해치므로 지양.

### `.claude_old/` — 시행착오 흔적 (참고용 보존)

저장소 루트의 `.claude_old/`는 **폐기된 초기 에이전트 팀 설정 스냅샷**. 현재 사용하는 `.claude/agents/` 구조로 오기 전의 실험 흔적으로, 재현성 참고용으로만 남겨둠.

**무엇이었나:**
- 이름: `bonsai-lane-solver` (팀 기반 multi-agent, 실험적 Agent Teams 기능 활용)
- 구성: `team-lead` (Sonnet) + `researcher-lit`/`researcher-data`/`researcher-model` (Haiku)
- 통신: SendMessage 기반 pull-first 원칙, 연구원별 inbox 파일
- 주제: 당시 plan.md v2.0 — "Bonsai 1-bit LLM + Lane + 솔버" (법률 도메인 pivot 이전)

**왜 폐기했나:**
- **30분 만에 토큰 소진**. SendMessage 기반 다자간 통신이 agent 간 왕복 오버헤드가 커서, 실제 연구 진전 대비 컨텍스트 비용이 과도.
- team-lead가 director 역할까지 겸임하는 구조에서 의례적 상태 체크가 빈발.
- Haiku 연구원의 품질 자가 점검이 "Sonnet 승격 요청"으로 이어져 비용 이중 발생.

**현재 구조와의 차이:**

| 항목 | `.claude_old/` (폐기) | `.claude/agents/` (현재) |
|------|----------------------|--------------------------|
| 통신 | SendMessage (pull inbox) | Agent tool 단발 호출, 결과 파일로 전달 |
| 오버헤드 | team-lead 상주 + 연구원 상주 | Director는 이 세션, 연구원은 필요 시 spawn |
| 비용 | 토큰 급격 소진 | 한 세션에서 R1+R2 완주 가능 |
| 상태 관리 | inbox JSON 파일 | TaskCreate/TaskUpdate + 파일 산출물 |

**유지 이유:**
- 초기 팀 프롬프트(team-lead, researcher-lit/data/model 각자의 역할 기술)에 **당시 설계 의도**가 남아 있어 참고 가치 있음.
- 향후 multi-agent 실험 재시도 시 "무엇이 비쌌는지" 비교 기준.

**주의:** `.claude_old/`는 **실행되지 않음**. Claude Code는 `.claude/` 만 인식. 삭제해도 무방하나 학습용으로 보존.

