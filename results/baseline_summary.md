# Baseline Results (A1, B1, C1)

**Date:** 2026-04-19
**Full LegalBench testset** (hearsay 94, PJ 50, rule_qa 50, textualism_tool_dictionaries 107)

## Matrix (balanced-accuracy / ROUGE-L)

| Task | A1 Qwen2.5-1.5B | B1 Qwen3-1.7B | C1 Qwen3 thinking |
|------|-----------------|----------------|--------------------|
| hearsay | **0.522** | 0.509 | 0.431 |
| personal_jurisdiction | 0.591 | 0.500 | **0.663** |
| rule_qa (ROUGE-L) | 0.121 | **0.139** | 0.113 |
| textualism_tool | **0.583** | 0.500 | 0.471 |
| **Mean** | **0.454** | 0.412 | 0.419 |

Parse rate: 100% across all cells.

## Observations

- Qwen2.5-1.5B가 Qwen3-1.7B CoT를 mean 기준 상회 (0.454 vs 0.412). 1.5B 주력 모델 선택 타당.
- Thinking 모드는 personal_jurisdiction에서만 +16.3pp로 큰 이득 (0.500 → 0.663). 다른 태스크에서는 오히려 하락.
- rule_qa는 전 모델에서 ROUGE-L 10~14% — Director 지시대로 L06 직접 생성으로 두어야 하며, SFT로 개선 여지 확인 필요.
- textualism_tool은 A1이 58%로 의외로 양호. Lane 재배치 근거(수사 분류) 여전히 유효하므로 L06 유지.

## Plan 예상치 vs 실측

| Task | Plan 예상 (1.5B 단독) | 실측 A1 | Delta |
|------|-----------------------|---------|-------|
| hearsay | 55~65% | 52.2% | -3~13pp |
| rule_qa | 50~60% | 12.1% | 크게 낮음 (메트릭 차이) |
| personal_jurisdiction | 50~60% | 59.1% | 범위 내 |
| textualism_tool | 45~55% | 58.3% | 상회 |

rule_qa는 balanced-accuracy가 아닌 ROUGE-L 사용해서 수치가 낮음. SFT 후 재검토.

## Step 7 결정

**GO.** SFT + DPO 진행. 근거:
1. baseline 파이프라인 완전 작동 (parse 100%).
2. hearsay 52% → 목표 80%+. personal_jurisdiction 59% → 75%+. Lane+Solver로 15~25pp 증가 목표.
3. A1이 B1보다 높음 — 솔버 이득 여유 충분.
