# R2 Final Analysis

**Date:** 2026-04-19
**Author:** Director

---

## 1. R2 vs R1 vs Baseline

| Task | A1 CoT | R1 A3 | **R2 A3** | R2 tool_call | R2 parse |
|------|--------|-------|-----------|--------------|----------|
| hearsay | 0.522 | 0.526 | **0.523** | 0.074 | 0.351 |
| personal_jurisdiction | **0.591** | 0.500 | 0.500 | 0.060 | 0.280 |
| rule_qa (ROUGE-L) | 0.121 | 0.092 | **0.117** | 0.160 | 1.000 |
| textualism_tool | **0.583** | 0.489 | 0.469 | 0.103 | 0.308 |
| **Mean** | **0.454** | 0.402 | **0.402** | — | — |

**핵심: R2 mean이 R1 mean과 동일 (0.402).** B1(0.412) 미달, A1(0.454) 미달.

### R2 A2 (SFT 없이 Outlines만)
- 모든 태스크에서 tool_call 0% — base model은 Outlines만으로는 Lane 프레임워크 활용 불가.
- parse_rate 5~8%로 매우 낮음 → 거의 "No" fallback으로 0.500 운 좋게 찍음.

## 2. Trace 분석 (실제 샘플 검증)

### 2.1 Pipeline 파싱 버그

가장 큰 발견. Hearsay 샘플 #0:
```
recorded_lane: L06   ← eval이 L06으로 잘못 인식
tool_call: null
final_answer: "<lane>L01</lane>\n<tool_call>{\"tool\":\"z3\",\"method\":\"check_elements\",...}</tool_call>\nAndrea, this is..."
```

**모델은 L01+tool_call을 제대로 생성했으나, `run_lane_solver_r2()`가 Outlines Lane choice와 별도로 free-generation을 돌려서 lane과 실제 텍스트가 엇나감.** tool_call은 실행 안 되고 최종 파서에서 전부 "No"로 fallback.

### 2.2 Tool_call이 발화된 샘플 — 내용이 엉터리

```json
{
  "elements": ["element_of_purpose", "element_of_malice"],
  "facts": ["neighbor stated Christopher's behavior was malicious", ...]
}
```
실제 hearsay 법리(FRE 801(c): "out-of-court statement offered for truth")와 무관한 영어 일반명사 환각. SFT 데이터가 한국어 법률 패턴에 과적합되어 영문 질문에서 generic한 elements를 만들어냄.

### 2.3 Final answer 품질

Binary 태스크에서 모델이 "Yes/No" 대신 paragraph를 뱉음 → parse_rate 28-35%. R2 설계 목표(final answer ≥80자)가 달성됐지만, 그 "자세한 답"이 정답 추출을 오히려 방해.

## 3. 실패 원인 정리

| # | 문제 | 증거 | 우선순위 |
|---|------|------|---------|
| 1 | Pipeline 파싱 버그 (tool_call 누출) | trace의 lane/tool_call 불일치 | **P0** |
| 2 | SFT 데이터 영문 커버리지 부족 | tool_call 발화 시 elements 환각 | **P0** |
| 3 | Binary 태스크에서 Yes/No 출력 강제 안 됨 | parse_rate 28-35% | **P1** |
| 4 | Outlines가 효과 있지만 pipeline에서 bypass됨 | A2 tool_call 0%, A3도 10% | **P1** |
| 5 | Reasoning 블록 부재 (L06) | rule_qa rouge 0.12 | **P2** |

## 4. R3 계획

### P0: Pipeline 버그 수정
- `run_lane_solver_r2()` 검사: Lane 선택과 tool_call 생성 사이에 문자열이 섞여 들어가는지.
- 출력 후 `final_answer`에 `<tool_call>` 문자열이 있으면 정규식으로 추출 + 재실행.
- 또는 `apply_chat_template`로 "assistant"가 `<lane>...</lane>`만 찍고 `\n`으로 끝나게 강제.

### P0: 영문 SFT 시드 확충
- LegalBench 스타일 hearsay 시드 200개 (FRE 801(c) 요건: `out_of_court_statement`, `offered_for_truth`)
- PJ 시드 200개 (`minimum_contacts`, `purposeful_availment`, `fair_play`)
- textualism 시드 200개 (binary classification 형식)
- 한국어 샘플 비중 축소: 70% → 30%

### P1: Binary 응답 강제
- Yes/No 태스크 전용 Lane (L01-binary) 신설.
- Outlines로 final answer도 `^(Yes|No)\\b.*` 강제.

### P1: Schema 엄격화
- `elements` 필드에 화이트리스트 강제 (예: hearsay는 801(c) 요건 2~3개만 허용).

### P2: Reasoning 블록 (L06 한정)
- `<reasoning>`/`</reasoning>` 블록을 tool 결과와 별도로 학습.
- rouge 개선 목적.

## 5. 판단

- R2가 가설을 **증명도 기각도 하지 못함** (R1과 동일 수치).
- Pipeline 버그 수정 + 영문 시드 확충이 R3의 최소 조건.
- 20:00 KST 기한까지 ~7.5시간 → R3 시도 1회 가능. 실패 시 이번 사이클 종료.
