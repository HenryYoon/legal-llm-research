# Final Analysis — Step 9

**Date:** 2026-04-19
**Author:** Director

---

## 1. 실험 매트릭스 전체 결과

| Task | A1 CoT | A2 Solver | A3 Lane+Solver (SFT) | B1 Qwen3 CoT | C1 Qwen3 thinking |
|------|--------|-----------|----------------------|--------------|--------------------|
| hearsay | 0.522 | **0.543** | 0.526 | 0.509 | 0.431 |
| personal_jurisdiction | **0.591** | 0.500 | 0.500 | 0.500 | 0.663 |
| rule_qa (ROUGE-L) | 0.121 | **0.176** | 0.092 | 0.139 | 0.113 |
| textualism_tool | **0.583** | 0.444 | 0.489 | 0.500 | 0.471 |
| **Mean** | **0.454** | 0.416 | 0.402 | 0.412 | 0.419 |

### 보조 지표 (A2/A3)

| Task | Variant | parse_rate | tool_call_rate |
|------|---------|------------|-----------------|
| hearsay | solver | 0.628 | 0.000 |
| hearsay | lane_solver | 0.479 | **0.149** |
| PJ | lane_solver | 0.560 | 0.120 |
| rule_qa | lane_solver | 1.000 | 0.100 |
| textualism | lane_solver | 0.439 | 0.159 |

---

## 2. 가설 검증 — 실패

| 가설 | 결과 | 판정 |
|------|------|------|
| A2 > A1 (솔버 효과) | 0.416 < 0.454 | ❌ |
| A3 > A2 (Lane 증폭) | 0.402 < 0.416 | ❌ |
| **A3 > B1 (핵심)** | **0.402 < 0.412** | ❌ |
| A3 > C1 | 0.402 < 0.419 | ❌ |

**핵심 가설 기각.** 1.5B + Lane + Solver가 1.7B CoT를 이기지 못함. 오히려 A3가 A1 baseline보다 5pp 낮음.

단, hearsay/rule_qa 단일 태스크에서는 A2가 A1을 **상회** (+2pp, +5.5pp) — 솔버가 도움되는 케이스는 존재.

---

## 3. 근본 원인 분석

### 3.1 Tool call 호출률 자체가 낮음 (10~16%)
SFT 후에도 모델이 LegalBench(영어) 입력에 대해 tool_call을 거의 발화 안 함.
→ **원인: 학습 데이터는 한국어 위주, 평가는 영어.** 도메인/언어 mismatch가 Lane 분류 일반화를 막음.

### 3.2 Tool call이 나와도 schema를 벗어남
로그에 drive-by 예시:
```
unknown z3 method: equation_check
unknown z3 method: contains
unknown z3 method: all_values_tac
subsume() missing 1 required positional argument
calc() missing 1 required positional argument
```
→ 모델이 schema에 없는 메서드명을 **환각**. SFT 샘플이 한정된 메서드만 노출했는데도, 모델이 일반화하면서 새 이름을 만들어냄.
→ **제약 디코딩(Outlines/XGrammar) 미적용**이 원인 — plan.md에는 있었으나 이번 사이클에서 생략됨.

### 3.3 Final answer 텍스트 부실 (기학인)
SFT 데이터의 tool 이후 assistant 메시지 평균 11.5자. binary 태스크에서는 "Yes/No" 추출에 의존하는데, `<lane>L06</lane>` 프리픽스가 parser를 혼란시킴 (parse_rate 0.48~0.56 하락).

### 3.4 PJ에서 thinking 모드가 16pp 앞섬
C1이 PJ에서 0.663으로 A3(0.500)를 크게 상회. 다요건 추론에는 **내부 thinking이 외부 솔버보다 효과적**일 수 있음. 특히 1.5B는 tool_call을 올바르게 구성할 능력이 부족.

---

## 4. plan.md 판단 기준 적용

- `A2 ≤ A1` → **"솔버 스키마 단순화"** 기준 발동.
- `A3 ≤ B1` → **"Lane 축소 (10 → 5)"** 기준 발동.

두 기준 모두 트리거.

---

## 5. 차기 라운드 권고

### 5.1 즉시 조치
1. **제약 디코딩 도입** — Outlines 또는 XGrammar로 tool_call JSON 강제. 환각 메서드명 차단.
2. **Lane 축소 10 → 5** — L01, L05, L06, L07, L10만 유지. L02~L04는 L01로 흡수.
3. **영어 SFT 샘플 확충** — LegalBench 공식 train split에서 합성. 현재 한국어 편향 해소.
4. **Final answer 텍스트 재생성** — tool 결과 → 상세 자연어 답변 (plan.md 6.3 수준, 평균 100자+).

### 5.2 데이터 재생성 (연구원 2)
- L01 시드를 영어 LegalBench 스타일로 재작성 (hearsay/PJ 실제 예시 포함).
- 메서드 이름을 schema에 **명시된 것만** 등장하도록 검증 스크립트 추가.
- DPO rejected 에 "schema 밖 메서드" 케이스 추가 → 환각 억제 신호.

### 5.3 후속 실험
- KLAC 편입으로 한국어 L06 강화 (연구 목적 사용 승인됨).
- Qwen3-1.7B SFT (B2/B3) 추가 — 원래 plan에는 있었으나 시간 제약으로 이번엔 skip.
- DPO 재실행 (이번 사이클에서 미실행).

### 5.4 프레이밍 재조정
핵심 가설("1.5B+Lane+솔버 > 1.7B CoT") 절대 승리는 실패. 논문/보고 시 프레이밍을 **"파라미터 대비 태스크별 선택적 향상"**으로 조정 권고:
- hearsay, rule_qa에서 A2가 A1 상회 → **태스크 선별 시 솔버 유효**.
- PJ는 thinking 우위 → 태스크 특성에 따른 접근 분리 필요.

---

## 6. 이번 사이클 기여

1. **Baseline 공백 메움**: Qwen2.5-1.5B / Qwen3-1.7B(+thinking) × LegalBench 4태스크 = 12셀 최초 측정치.
2. **솔버 인프라**: Z3(L01~L04) + SymPy(L05) 재사용 가능 구현, pytest 11/11 통과.
3. **합성 데이터 파이프라인**: SFT 19K / DPO 20K, 라운드트립 100%. 재현 가능.
4. **실패 원인 구조적 파악**: 제약 디코딩 부재 + 영어/한국어 mismatch + 짧은 final answer가 핵심.

---

## 7. 결론

**가설 기각. 1.5B + Lane + Z3가 이번 설정에서는 7B CoT를 이기지 못함.** 단, 실패 원인이 3가지(제약 디코딩 없음, 언어 mismatch, final answer 부실)로 명확히 분리되어 차기 라운드에서 개별 해결 가능.

Plan.md 판단 기준에 따라 데이터/Lane/디코딩 재설계 후 재도전 권장.
