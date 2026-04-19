# Legal Lane + Solver

작은 LLM(1.5B)에 법률 질문 유형을 분류하도록 가르치고, 논리 판단은 **Z3 솔버**에게 떠넘겨서 7B급 모델을 이길 수 있는지 실험하는 저장소.

- 모델: Qwen2.5-1.5B-Instruct, Qwen3-1.7B
- 평가: LegalBench 4개 태스크 (hearsay, personal_jurisdiction, rule_qa, textualism_tool)
- 환경: RTX 3060 12GB

---

## 아이디어

법률 질문에 답하려면 보통 3단계가 필요함:

1. 사실 읽기 ("누가 뭘 했다")
2. 요건 찾기 ("절도죄 요건은 타인의 재물 + 절취")
3. **요건이 충족되었는지 논리 판정**

1.5B는 1·2는 웬만큼 하는데 3에서 자주 엉킴. 그래서:

- 모델 역할: **질문 유형(Lane) 분류 + 요건·사실을 JSON으로 정리**
- Z3 역할: **정리된 JSON을 받아 Boolean 판정**

### Lane 예시

```
user: "편의점에서 음료를 주머니에 넣고 나왔다. 절도죄?"

모델: <lane>L01</lane>
      <tool_call>{
        elements: ["타인의 재물", "절취"],
        facts:    ["편의점 음료", "주머니에 넣고 퇴장"],
        matching: {"타인의 재물": true, "절취": true}
      }</tool_call>

Z3:   {all_met: true}

모델: "절도죄 성립. 두 요건 충족."
```

설명/요약처럼 논리 필요 없는 질문은 그냥 텍스트로 답함.

### 현재 Lane 5개 (R2 이후)

| Lane | 용도 | 처리 |
|------|------|------|
| L01 | 요건 매칭 | Z3 |
| L05 | 수치 계산 | SymPy |
| L06 | 설명·비교·요약 | 모델 직접 |
| L09 | 법률 용어 번역 | 모델 직접 |
| L10 | 정보 부족 | 모델 직접 |

> Lane 개념은 [DinoDS AI — Lanes](https://dinodsai.com/lanes)의 "Lane = schema + QA contract + 결정적 출력"을 법률 도메인에 맞게 축소/재구성한 것.

---

## 실험 매트릭스

|  | CoT만 | +Solver | +Lane+Solver |
|--|-------|---------|--------------|
| Qwen2.5-1.5B | A1 | A2 | **A3** |
| Qwen3-1.7B | B1 | B2 | B3 |
| Qwen3-1.7B thinking | C1 | — | — |

**핵심 가설**: A3 > B1. 즉 1.5B + Lane + Solver가 1.7B CoT를 이긴다.

---

## 진행 상황

| 라운드 | 내용 | 결과 |
|--------|------|------|
| R1 | 첫 번째 시도. Lane 10개, 한국어 중심 SFT | **A3 mean 0.402, 기각** [분석](results/analysis.md) |
| R2 | Lane 5개 축소, 영어 70%, Outlines 제약 디코딩 | A3 mean **0.402** (변화 없음) [분석](results/analysis_r2.md) |
| R3 | Pipeline 버그 수정, 영문 시드 600+ 확충, binary regex 강제 | 🔄 진행 중 |

### R1/R2 결과

| Task | A1 | B1 | C1 | R1 A3 | R2 A3 |
|------|----|----|----|-------|-------|
| hearsay | 0.522 | 0.509 | 0.431 | 0.526 | 0.523 |
| personal_jurisdiction | 0.591 | 0.500 | 0.663 | 0.500 | 0.500 |
| rule_qa (ROUGE-L) | 0.121 | 0.139 | 0.113 | 0.092 | 0.117 |
| textualism_tool | 0.583 | 0.500 | 0.471 | 0.489 | 0.469 |
| **평균** | **0.454** | 0.412 | 0.419 | 0.402 | 0.402 |

### R2 실패 원인

1. **Pipeline 파싱 버그** — `<tool_call>...` 문자열이 final_answer로 새어 lane=L06으로 오인됨.
2. **영문 시드 부족** — 한국어 SFT에 과적합되어 영어 LegalBench에서 `element_of_purpose` 같은 헛소리 요건 생성.
3. **Binary 응답 포맷 없음** — Yes/No 물음에 paragraph로 답해서 parse_rate 28~35%.

R3에서 이 셋을 직접 공략 중.

---

## 디렉토리

```
llm-research/
├── CLAUDE.md, README.md        루트 필수 2개
├── docs/plan.md                연구 계획 v4
├── context/                    Director → 연구원 지시 (R1/R2/R3)
├── solver/                     Z3 + SymPy 래퍼, pytest
├── data/
│   ├── seeds/, seeds_r2/, seeds_r3/   Lane별 시드
│   ├── sft_*.jsonl, dpo_*.jsonl        학습 데이터
│   └── legalbench/                      평가 CSV
├── scripts/                    build/generate/validate/train/eval
├── configs/                    sft/dpo × qwen25/qwen3 × r1/r2/r3
├── reports/                    literature_review, data_quality*
├── results/                    baseline/experiment/analysis
└── .claude/agents/             researcher1/2/3 정의
```

`models/`는 용량 문제로 git 제외.

---

## 실행

```bash
uv venv .venv && source .venv/bin/activate
uv pip install transformers accelerate bitsandbytes peft trl datasets \
               z3-solver sympy jsonschema pyyaml pandas scikit-learn \
               rouge-score outlines

# 솔버 테스트
python -m pytest solver/tests/

# 데이터 생성
python scripts/build_seeds.py
python scripts/generate_data.py        # 또는 _r2, _r3

# Baseline
python scripts/eval_baseline.py --model qwen25 --task all

# SFT
python scripts/train_sft.py --config configs/sft_qwen25_r3.yaml

# 평가
python scripts/eval_with_solver_r3.py --model qwen25 --all-tasks \
       --adapter models/qwen25-1.5b-legal-sft-r3 \
       --max-samples 200 --trace results/r3_a3_trace.jsonl
```

---

## 팀 구성

Claude Code multi-agent로 진행:

| 역할 | 모델 | 담당 |
|------|------|------|
| Director | Opus 4.7 | 총괄, 리뷰, context 번역, 최종 분석 |
| 연구원 1 | Sonnet | 선행연구 조사 |
| 연구원 2 | Sonnet | 솔버 + 데이터 생성 |
| 연구원 3 | Haiku | 학습/평가 스크립트 + 실행 |

각 연구원은 Agent tool로 일회성 spawn, 결과는 파일로 전달.

---

## 알려진 제약

- SFT 1 epoch: ~1시간 (Unsloth 미사용, transformers+PEFT)
- Qwen3 thinking 평가: CoT 대비 5배 느림
- KLAC 한국어 데이터: 라이선스 명시 부재로 현재 미사용
- R1~R3 솔버 코드가 병행 관리되어 중복 있음 (R3 종료 후 `SolverProtocol`로 리팩터 예정)

---

## 데이터 / 참고

- [LegalBench](https://huggingface.co/datasets/nguha/legalbench) (CC-BY-4.0)
- [KLAC](https://huggingface.co/datasets/jihye-moon/klac_legal_aid_counseling) — 보류
- 합성 데이터: subagent가 Python으로 생성 (외부 API 없음)

**선행연구**:
- SatLM (NeurIPS 2023), Logic-LM (EMNLP 2023), NL2Logic (2026) — NL → Z3/FOL 파이프라인
- SaulLM (NeurIPS 2024) — 540B 법률 토큰 pretrain, 파라미터 효율 비교군
- DinoDS AI — Lanes ([링크](https://dinodsai.com/lanes)) — Lane 개념 원천
- LegalBench 논문 (arXiv 2308.11462)

상세는 [`reports/literature_review.md`](reports/literature_review.md).

---

## 라이선스

연구용 (학술 비상업). 데이터셋은 각 출처 라이선스 준수.

---

## 부록: 저장소 이전 이력

이 저장소는 `fb38ebc` 커밋(2026-04-19)으로 초기화됨. 그 이전의 편집은 Claude Code 로컬 `~/.claude/file-history/`에 스냅샷만 있어 git으로 재구성 불가.

요약 타임라인:

| 시기 | 내용 |
|------|------|
| 2026-04-08 ~ 17 | 탐색 — Bonsai 1-bit LLM + 솔버 주제로 시작 |
| 2026-04-17 ~ 18 | 법률 도메인으로 pivot, plan.md v4 확정 |
| 2026-04-19 04:20 | 현재 세션 시작 — Step 1~R3 자동 실행 |
| 2026-04-19 06:11 | `fb38ebc` 초기 commit |

### `.claude_old/` — 폐기된 multi-agent 시도

팀 기반 multi-agent(`bonsai-lane-solver`)를 먼저 시도했으나 **30분 만에 토큰 소진**. SendMessage 왕복 오버헤드가 실제 진전 대비 컸음. 현재는 Agent tool 단발 호출 방식(`.claude/agents/`)으로 전환. 학습용으로만 `.claude_old/` 보존, **실행되지 않음**.
