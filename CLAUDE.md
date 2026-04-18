# Legal Lane + Solver Research

## 연구 목표
법률 추론에서 1.5B 모델 + Lane + Z3 솔버가 7B CoT를 이기는 것을 증명.

## 핵심 가설
법률 추론을 Lane 구조로 분류하고 형식화 가능한 추론을 Z3 솔버로 외주하면,
1.5~1.7B 모델이 법률 벤치마크에서 7B+ CoT 모델을 이길 수 있다.

## 팀
- **Director (opusplan)**: 연구 총괄, 리뷰, 맥락 번역
- **연구원 1 (sonnet)**: 선행연구 — `.claude/agents/researcher1.md`
- **연구원 2 (sonnet)**: 데이터/솔버 — `.claude/agents/researcher2.md`
- **연구원 3 (haiku)**: 모델링 — `.claude/agents/researcher3.md`

## 환경
- GPU: RTX 3060 12GB
- Python 3.10+
- `pip install unsloth trl z3-solver sympy outlines datasets`

## 베이스 모델
- Qwen2.5-1.5B-Instruct (주력)
- Qwen3-1.7B (비교군)

## 실행 순서
1. 연구원 1 → 선행연구 조사 → `reports/literature_review.md`
2. Director 리뷰 → `context_for_researcher2.md`
3. 연구원 2 → 솔버 + 데이터 → `solver/`, `data/`, `reports/data_quality.md`
4. Director 리뷰 → `context_for_researcher3.md`
5. 연구원 3 → baseline (A1, B1, C1)
6. Director 리뷰
7. 연구원 3 → SFT + DPO (GPU 전용)
8. 연구원 3 → 실험군 (A2, A3, B2, B3)
9. Director → 최종 분석 → `results/analysis.md`

## VRAM 제약
- 학습 시 GPU 독점
- 추론 4bit 양자화 필수
- batch 4, OOM 시 2로 축소

## 데이터
- LegalBench: `nguha/legalbench` (HuggingFace)
- 한국어 법률 상담 QA: `jihye-moon/klac_legal_aid_counseling` (HuggingFace)
- 합성 데이터: subagent 직접 생성, 별도 API 호출 없음

## Lane 체계
**형식화 (솔버 경유)**: L01 요건매칭, L02 규칙적용, L03 포섭, L04 논리판단, L05 계산
**직접 생성**: L06 설명, L07 사례비교, L08 요약, L09 번역, L10 불확실

## 판단 기준
- 데이터 라운드트립 통과율 < 85% → 수정
- A2 ≤ A1 → 솔버 스키마 단순화
- A3 ≤ B1 → Lane 축소 (10 → 5)
- 동일 선행연구 발견 → 차별화 재설정
