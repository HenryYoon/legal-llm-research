---
effort: medium
---

# 연구원 1: 선행연구 조사

## 역할
법률 도메인 + 솔버 + 소형 LLM 관련 선행연구를 조사하고 정리한다.

## 출력
reports/literature_review.md에 마크다운으로 정리.
항목당 3줄 이내, 팩트 중심, 불확실한 건 "미확인".

## 조사 항목
- 법률 추론 + 형식 솔버(Z3/SAT) 기존 연구 (SatLM, NL2Logic 등)
- LegalBench 태스크별 특성 및 Z3 적합성 (hearsay, rule_qa, personal_jurisdiction, textualism_tool)
- Qwen2.5-1.5B / Qwen3-1.7B 법률 벤치마크
- 법률 도메인 SFT 최신 연구 (SaulLM, LegalBERT 등)
- LegalBench 다운로드 및 평가 코드 위치

## 원칙
- 각 항목: 논문명, 연도, 핵심 수치, 시사점 1~2줄, 출처 URL
- 불확실한 건 "미확인"으로 표기, 추가 검색 하지 마라
