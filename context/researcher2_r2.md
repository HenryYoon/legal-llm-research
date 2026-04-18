# Context for 연구원 2 — Round 2 (데이터 재생성 + 제약 디코딩)

**From:** Director
**Date:** 2026-04-19
**Based on:** results/analysis.md (R1 가설 기각)

---

## 1. R1 실패 요약

- A3 mean 0.402 < B1 0.412 < A1 0.454. 핵심 가설 기각.
- 근본 원인 3가지:
  1. **Tool call 호출률 10~16%** — 한국어 SFT / 영어 평가 mismatch.
  2. **Schema 밖 메서드 환각** — `equation_check`, `all_values_tac` 등 없는 메서드 호출.
  3. **Final answer 평균 11.5자** — binary 태스크 parse_rate 하락, 생성형 태스크 rouge 낮음.

## 2. R2 재설계 지시

### 2.1 Lane 10 → 5 축소 (plan.md 판단 기준 트리거)

| 신규 Lane | 설명 | 통합 전 |
|-----------|------|---------|
| **L01 요건 매칭** | Boolean AND/OR, 요건-사실 매칭 | L01+L02+L03+L04 전부 흡수 |
| **L05 계산** | SymPy 수치 | L05 |
| **L06 설명** | 자유 생성 (설명/비교/요약) | L06+L07+L08 통합 |
| **L09 번역** | 법률 용어 한↔영 | L09 |
| **L10 불확실** | 판단 불가 | L10 |

- L02/L03/L04는 전부 L01로 흡수. 실무에서 구분 모호하고 분류 노이즈 발생.
- 모든 Z3 호출은 `check_elements` 한 메서드로 통일 (mode=and/or, 필요 시 nested).
- tool 종류 2개로 축소: `z3.check_elements`, `sympy.calc`.

### 2.2 제약 디코딩 도입 (최우선)

- **Outlines** 라이브러리 사용. `uv pip install outlines`.
- tool_call JSON schema를 Outlines `json_schema` 로 모델 생성 시 강제.
- 평가 스크립트(`eval_with_solver.py`)에서 `<tool_call>` 블록 내부만 제약 적용 (두 단계 생성 또는 stop token 활용).
- 대안: `outlines` 설치 실패 시 `xgrammar` 또는 `guidance`. 둘 다 실패하면 **regex + retry** fallback.
- Lane 분류도 제약: `<lane>(L01|L05|L06|L09|L10)</lane>` 토큰 강제.

### 2.3 SFT 데이터 재생성

**언어 구성 변경:**
- 영어 70% (LegalBench 스타일) + 한국어 30%.
- LegalBench train split에서 hearsay(5), PJ(4), textualism(4) 소량이라도 시드에 포함. 이를 템플릿 변형으로 영문 증강 2K 생성.

**Schema 엄격 검증:**
- 생성된 모든 tool_call이 `{"tool": "z3", "method": "check_elements", ...}` 또는 `{"tool": "sympy", "method": "calc", ...}` **정확히 일치**할 것.
- 스키마 밖 메서드 생성 시 해당 샘플 **폐기**. DPO rejected에는 schema 밖 메서드 케이스 추가.

**Final answer 길이 상향:**
- tool 실행 후 assistant 메시지 **평균 80자 이상**. 최소 30자.
- 포맷: "[결론]. [근거 조문/규칙]. [핵심 요건 충족 여부]."
- plan.md 6.3 예시 수준 준수.

**목표 규모:**
- SFT: L01 4K (영어 3K + 한국어 1K), L05 1K, L06 3K, L09 500, L10 500 → 총 9K (R1의 19K보다 작게, 품질 우선).
- DPO: 10K 쌍. Rejected 구성:
  - 50% schema 밖 메서드 환각
  - 30% 잘못된 elements/facts 매칭
  - 20% final answer 부실 (짧거나 한 줄 템플릿)

### 2.4 솔버 정리

`solver/z3_legal.py`:
- `check_elements(elements, facts, matching, mode)` 하나만 export. L02/L03/L04 함수들은 내부 구현으로만 유지 (하위 호환 위해 남겨두되 schema에선 제외).
- `schemas.py` 재작성: Lane 5개, tool 2개, 나머지 전부 삭제.
- `validator.py`: 제약 디코딩 전제로 로직 단순화. schema 밖 메서드는 validation fail로 처리.

## 3. 체크포인트

1. `solver/schemas.py` 재작성 → pytest 통과.
2. 시드 영어 100개 + 한국어 30개 작성 후 Director에게 샘플 10개 보고 (라운드트립 통과 확인).
3. 증강 스크립트 메서드 검증 로직 추가 후 전체 재생성.
4. `reports/data_quality_r2.md`: 언어 분포, final answer 평균 길이, schema 통과율, tool 종류 분포.
5. Outlines 통합 smoke test (single inference) 성공 후 eval 스크립트 반영.

## 4. 출력 산출물

- `solver/` 정리 (R2 schemas 반영)
- `data/seeds_r2/` 새 시드
- `data/sft_r2.jsonl`, `data/dpo_r2.jsonl`
- `scripts/generate_data_r2.py`
- `scripts/eval_with_solver.py` — Outlines 통합 패치
- `reports/data_quality_r2.md`

## 5. 완료 보고 포맷 (5~8줄)

- schema 통과율, 언어 분포, final answer 평균 길이
- Outlines smoke test 성공/실패 및 대안
- 총 샘플 수, 이슈/블로커
- Step R2-3(SFT 재학습) 진행 준비 여부

---

**중요:** R1 데이터/모델은 그대로 두고 `_r2` suffix로 병행 관리. 비교 분석을 위함.
