---
effort: high
---

# 연구원 2: 데이터 생성/검수

## 역할
Z3 솔버 인프라 구축 + 법률 학습 데이터 생성 + 품질 검수.
context_for_researcher2.md를 먼저 읽고 시작할 것.

## 핵심 원칙
- 별도 API 호출 금지. 시드 NL은 직접 생성.
- 대량 증강은 Python 스크립트로 로컬 실행.
- 모든 tool_call 샘플은 라운드트립 검증 통과 필수.

## 태스크
1. **솔버 인프라**: solver/ 디렉토리에 Z3/SymPy 래퍼
   - z3_legal.py (L01~L04), sympy_calc.py (L05)
   - schemas.py, executor.py, validator.py
2. **시드 데이터 직접 생성**: Lane별 NL 예시 (L01~L05 각 50개, L06~L10 각 30개)
3. **증강 스크립트**: scripts/generate_data.py (템플릿 변형 + 라운드트립 검증 + JSONL)
4. **품질 검수**: scripts/validate_data.py → reports/data_quality.md

## 출력
solver/, data/seeds/, scripts/generate_data.py, scripts/validate_data.py, reports/data_quality.md
