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

## 태스크
1. **학습**: scripts/train_sft.py, scripts/train_dpo.py (--model qwen25 | qwen3)
2. **평가**:
   - scripts/eval_baseline.py (A1, B1, C1)
   - scripts/eval_with_solver.py (A2, A3, B2, B3)
   - 대상: LegalBench (hearsay, rule_qa, personal_jurisdiction, textualism_tool), klac
3. **결과 집계**: results/experiment_matrix.csv, comparison_table.md, analysis.md

## 솔버 연동 평가 로직
1. tool_call 생성 프롬프트 → 2. 파싱 → solver/executor.py → 3. 정답 비교
4. 파싱 실패 시 fallback: 모델 직접 답변
