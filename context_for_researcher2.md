# Context for 연구원 2 (데이터/솔버)

**From:** Director
**Date:** 2026-04-18
**Based on:** reports/literature_review.md

---

## 1. 선행연구 요약 (너에게 필요한 부분만)

- **SatLM/Logic-LM/NL2Logic**: NL → SMT-LIB/Z3 파이프라인은 검증됨. LSAT/FOLIO까지 다뤘으나 **LegalBench 적용은 공백**.
- **NL2Logic(2026)**: 0.5B 모델에서도 solver grounding 작동 — 우리 1.5B 타깃 근거.
- **SaulLM-54B/141B**: 540B 법률 토큰 pretraining. 우리는 파라미터-효율 프레이밍으로 차별화.
- **LegalBench**: `nguha/legalbench` HF, 공식 평가 `HazyResearch/legalbench` GitHub. balanced-accuracy 기본, 태스크별 prompt template 포함.

## 2. Lane 재배치 지시 (중요)

plan.md의 Lane 매핑을 다음과 같이 **수정**하라:

| 태스크 | 기존 | 수정 | 근거 |
|-------|------|------|------|
| hearsay | L01 | **L01 유지** | 2~3 요건 Boolean AND — Z3 교과서 케이스 |
| personal_jurisdiction | L01 | **L01 유지** | 다요건(minimum contacts, purposeful availment, fair play) AND/OR |
| rule_qa | L02 | **L06 (직접 생성)** | rule-recall(암기) 태스크 — 형식화 대상 자체 없음 |
| textualism_tool | L03 | **L06 (직접 생성)** 또는 제외 | 수사(rhetoric) 분류 — 형식화 이득 제한적 |

**결론: Z3 수혜 태스크는 hearsay + personal_jurisdiction 2개에 집중.** 데이터 증강도 이 두 태스크의 요건 구조를 우선.

## 3. 솔버 인프라 우선순위

1. **L01 요건 매칭 (Z3)** — 최우선. hearsay/personal_jurisdiction에서 직접 사용.
   - 스키마: `{elements: [str], facts: [str], matching: {element: bool}, all_met: bool}`
   - Z3 구현: Boolean 변수 + And/Or + check().
2. **L02 규칙 적용 (Z3)** — 조문 conditional rule. rule_qa에는 사용 안 하지만 합성 데이터로 학습 가치 있음.
3. **L03 포섭 (Z3)** — L01과 구조 유사. 중복 회피, 얇게 구현.
4. **L04 논리 판단 (Z3)** — Logic-LM 패턴 참고. NL2FOL fallback 전략 포함.
5. **L05 계산 (SymPy)** — 손해배상/이자/기간. 독립 모듈.

## 4. 데이터 생성 방침

- 시드 NL: plan 그대로 Lane당 50 × 10 = 500개 **직접 작성**. 별도 API 금지.
- **라운드트립 우선**: NL → tool_call(JSON) → Z3 → 결과. 검증 통과율 **85%+ 필수**.
- Lane 분포: L01 비중을 기존 2K에서 **3K로 증강** (hearsay/PJ 집중 반영). L02/L03/L04/L05는 각 1.5K로 축소.
- KLAC 데이터는 **로딩 후 스키마 확인 먼저**. 라이선스 확인 불가 시 L06 SFT에서 제외.

## 5. LegalBench 평가 준비

- HazyResearch/legalbench GitHub에서 **hearsay, personal_jurisdiction, rule_qa, textualism_tool** 4개 태스크의 CSV + prompt template을 data/legalbench/ 아래 복사.
- 평가는 balanced-accuracy로 통일.

## 6. 체크포인트

- solver/ 작성 후 `pytest solver/tests/` 또는 자체 검증 스크립트로 Z3 기본 동작 확인 보고.
- 시드 500개 완료 후 라운드트립 통과율을 reports/data_quality.md에 기록.
- 통과율 < 85% → 스키마 단순화 후 재작성.

## 7. 출력 산출물

- `solver/` (__init__, z3_legal, sympy_calc, schemas, executor, validator)
- `data/seeds/` Lane별 JSON
- `data/legalbench/` 4개 태스크 CSV
- `scripts/generate_data.py`, `scripts/validate_data.py`
- `reports/data_quality.md`

완료 후 Director에게 5줄 이내 요약 보고.
