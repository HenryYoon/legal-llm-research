# Context for 연구원 3 (모델링)

**From:** Director
**Date:** 2026-04-18
**Based on:** reports/data_quality.md, context_for_researcher2.md

---

## 1. 현재 상태

- 솔버 인프라: `solver/` 완료 (Z3 L01~L04, SymPy L05). `python -m pytest solver/tests/` 11/11 pass.
- SFT 데이터: `data/sft_merged.jsonl` 19K (L01 3K, L02~L05 각 1.5K, L06~L10 각 2K). 라운드트립 100% 통과.
- DPO 데이터: `data/dpo_pairs.jsonl` 20K (chosen pass, rejected fail).
- LegalBench: `data/legalbench/` CSV 준비됨 (hearsay 94 test, PJ 50 test, rule_qa 50 test, textualism_tool_dictionaries 107 test).
- KLAC: 라이선스 공란으로 **L06 SFT에서 제외**. 한국어 평가는 보류.
- 환경: `uv venv .venv` 사용. 스크립트 실행 전 `source .venv/bin/activate` 필요.

## 2. Lane 재배치 (반드시 반영)

- Z3 수혜 태스크: **hearsay, personal_jurisdiction** (L01).
- rule_qa, textualism_tool은 **L06 직접 생성**. 솔버 경로 사용하지 마라.

## 3. 평가 설계

- 벤치마크: hearsay, personal_jurisdiction, rule_qa, textualism_tool (4개).
- 지표: **balanced-accuracy** 통일 (LegalBench 공식 지표).
- 프롬프트: HazyResearch/legalbench GitHub의 task 폴더 prompt template을 사용하거나 최소 재현 프롬프트 사용.
- 실험 셀:
  - A1 Qwen2.5-1.5B CoT only
  - A2 Qwen2.5-1.5B + 솔버 (Lane 없이 무조건 tool_call 시도)
  - A3 Qwen2.5-1.5B + Lane + 솔버 (SFT 후)
  - B1/B2/B3 동일 Qwen3-1.7B
  - C1 Qwen3-1.7B thinking 모드

## 4. 베이스라인 우선 (Step 5)

**지금 단계에서는 학습 금지.** baseline A1/B1/C1만 측정.

- `scripts/eval_baseline.py --model qwen25 --task hearsay` 형태로 인자 받기.
- 4bit 양자화 필수 (bitsandbytes or unsloth). 12GB VRAM 제약.
- 결과: `results/baseline.csv` (model, task, accuracy, n, pass_rate_parse).
- A1/B1/C1 수치가 선행연구(LegalBench 수치 미확인)와 비교 가능하도록 로그 상세화.

## 5. 학습 파이프라인 (Step 7, Director 승인 후)

- `scripts/train_sft.py --config configs/sft_qwen25.yaml` — Unsloth + TRL.
- `scripts/train_dpo.py --config configs/dpo_qwen25.yaml` — SFT 체크포인트 기반.
- OOM 시 자동 fallback: batch 절반 + grad_accum 2배. 로그 남겨라.
- LoRA 어댑터로 저장 (`models/qwen25-1.5b-legal-sft/`, `-dpo/`).

## 6. 실험군 평가 (Step 8)

- `scripts/eval_with_solver.py` — 모델에 tool_call 프롬프트 → `<tool_call>` 파싱 → `solver/executor.py` dispatch → 최종 답.
- 파싱 실패 fallback: 모델 직접 답변.
- 결과: `results/experiment_matrix.csv`, `results/comparison_table.md`.

## 7. 주의사항

- **데이터 다양성 리스크**: 증강이 boolean 반전 + 템플릿 변형 위주 → 실 LegalBench 일반화 우려. A3 < B1 나오면 Director에게 보고. 데이터 재생성 전 스크립트만 완성해둘 것.
- DPO rejected가 전부 roundtrip fail → 학습 신호 단순. 학습 후 DPO 효과 미미하면 Director에게 보고.
- LegalBench의 textualism은 `textualism_tool_dictionaries` variant를 사용함. 평가 코드에서 이 variant명으로 로딩.

## 8. 출력 산출물 (Step 5 단계)

- `scripts/eval_baseline.py`
- `scripts/eval_with_solver.py` (스켈레톤까지)
- `scripts/train_sft.py`, `scripts/train_dpo.py` (스켈레톤, 실행은 Step 7에서)
- `results/baseline.csv` (A1, B1, C1 4태스크 × 3모델 = 12 셀)
- 완료 후 Director에게 수치 5~8줄 요약 보고.

실행 환경: `source .venv/bin/activate && python scripts/eval_baseline.py ...`
