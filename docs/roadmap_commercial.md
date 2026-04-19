# Commercial Roadmap

**작성:** 2026-04-19 (R3 완료 전, pivot 결정 시점)
**적용:** 2026-04-20 세션부터

---

## 목적

R1~R3는 **"1.5B + Lane + Z3 > 1.7B CoT"** 가설을 검증하는 PoC. 실제 법률 서비스로 쓰려면 기준이 다른 종목이므로, 다음 사이클부터는 설계 전반을 상용화 수준으로 재설정한다.

## 기준 전환

| 항목 | 현재 (PoC) | 상용화 |
|------|------------|--------|
| 태스크별 정확도 | > B1 (0.412) | **0.85+** |
| LegalBench 커버 | 4 | **최대 10개, 깊이 있게** |
| 환각 | 허용 (trace로 확인) | **사실상 0**, 조문 인용 강제 |
| 거부 | L10 태그만 | 능동적 "모름" + "변호사 상담" |
| 언어/관할 | 영문 중심 | 한·영 이중 + 관할권 태그 |
| 감사 | trace JSON | 각 판단에 조문 링크 + 재현 가능성 |
| 모델 | 1.5B 주력 | **7B+** 주력, 1.5B는 라우터 역할만 |
| 지연 | ~7초/샘플 | < 1초 (스트리밍) |

## 구조 변경

### 1. RAG 도입 (최우선)

시드에 요건(`out_of_court_statement` 등)을 **하드코딩하지 않음**. 대신:

```
user 질문 → 검색 모듈(BM25 + 법령 임베딩) → 관련 조문/판례 top-k
         → Lane 분류 (요건은 RAG 결과에서 추출)
         → tool_call 생성 (elements = RAG에서 자동 유도)
         → Z3 판정
         → 조문 인용 포함 final answer
```

**필요 리소스:**
- 법령 코퍼스: 한국법제처 + US Code + LegalBench 인용 판례
- 임베딩 모델: multilingual-e5 또는 BGE-M3
- 벡터 DB: Qdrant or sqlite-vec

### 2. 거부 학습 강화

- L10 비중 현재 3% → **15~20%**
- "모름"이 정답인 시나리오 템플릿 확대
- reward: 틀린 답 > 거부 > 맞는 답 순으로 페널티 설계

### 3. 인용 강제

`solver/schemas_r3.py`의 tool_call schema에 필수 필드 추가:

```json
{
  "elements": [...],
  "facts": [...],
  "source_statutes": ["Fed. R. Evid. 801(c)", "..."],  // 신규 필수
  "source_cases": ["International Shoe v. Washington"]  // 선택
}
```

Outlines enum에서 실제 존재하는 조문 ID만 허용 → 환각 원천 차단.

### 4. 다단계 검증

```
Model A (Qwen-7B + Lane) → 답변 안
Model B (다른 모델, 예: Mistral-7B) → 동일 질문 독립 답
일치 → 반환
불일치 → "확신 없음" 거부
```

비용은 2배지만 상용화엔 필수.

### 5. 모델 업그레이드

| 역할 | 모델 후보 | 용도 |
|------|-----------|------|
| 주력 | Qwen2.5-7B-Instruct 또는 SaulLM-7B | Lane 분류 + tool_call 생성 |
| 라우터 | Qwen2.5-1.5B (R3 체크포인트 재사용) | 입력 → 주력/거부 분기만 |
| 검증 | Mistral-7B-Instruct | 독립 답변 생성 |

RTX 3060 12GB에서 7B 4bit SFT는 LoRA r=8로 가능. 다만 시간은 R3의 3~4배.

## 베이스라인 재측정

상용화 트랙에서는 1.5B baseline 재사용 불가. 새로 측정:

| 실험 | 모델 | 조건 |
|------|------|------|
| A1' | Qwen2.5-7B CoT | 선별된 10 태스크 이내 |
| A2' | +RAG | 조문 검색 결합 |
| A3' | +Lane+Z3+RAG | 완전형 |
| B'  | SaulLM-7B | 법률 특화 비교 |

**목표: A3' 각 태스크 0.85+, 평균 0.85+**.

### 태스크 선정 원칙 (최대 10개)

범위를 좁혀 **깊이**를 확보. 30+ 태스크를 얕게 다루면 상용 가치 0. 선정 기준:

1. **L01 요건 매칭 강점군** (Z3 이득 명확) — hearsay, personal_jurisdiction, diversity_jurisdiction, corporate_lobbying, contract_nli_* (3개 정도)
2. **L06 설명/비교 실전 수요** — rule_qa 일부, contract_qa, citation_prediction
3. **L09 번역 / 관할 용어** — 한↔영 법률 용어 매칭 (KLAC + 영문 대응)

선정 후보 예시 (10개):
1. hearsay
2. personal_jurisdiction
3. diversity_jurisdiction
4. contract_nli_no_solicitation
5. contract_nli_limited_use
6. corporate_lobbying
7. rule_qa (핵심 법 규칙 일부)
8. contract_qa
9. citation_prediction_open
10. cuad_audit_rights

**제외**: textualism 계열(수사 분류, 솔버 이득 적음), 다수의 open-ended generation (평가 노이즈 큼).

> 실제 선정은 Qwen-7B baseline 측정 후 "솔버가 유의미한 개선 가능한 태스크"를 우선 기준으로 최종 확정.

## 데이터

- 기존 9K 합성 시드 대부분 폐기. 합성 데이터는 패턴 학습용으로만.
- **실제 LegalBench train split** (소량이지만 진짜 데이터) 활용 우선.
- 한국어: KLAC 법률 상담 (라이선스 재확인 후) + 대법원 판례 요지.
- 스키마 검증: 조문 번호 + 판례 이름은 실제 DB 매칭 통과한 것만.

## 프레이밍 (논문/공개 시)

- 기존: "1.5B로 7B를 이긴다" — 과장이고 틀림
- 신규: "**Lane 라우팅 + RAG + 솔버 하이브리드가 LegalBench에서 7B 전용 모델과 동등하며, 추론 비용을 X% 절감한다**"

상용화 가치는 "SOTA 능가"가 아니라 "SOTA 수준을 저비용으로"에 있음.

## 일정 (초안)

| 단계 | 기간 | 산출물 |
|------|------|--------|
| S1 | ~2026-05-01 | RAG 인프라 + 법령 코퍼스 인덱싱 |
| S2 | ~05-15 | Qwen-7B baseline (A1') + 데이터 재설계 |
| S3 | ~05-31 | SFT (+RAG) — A2' 측정 |
| S4 | ~06-15 | Lane + Z3 통합 — A3' 측정 |
| S5 | ~06-30 | 다단계 검증 + 거부 평가 |
| S6 | ~07-15 | 10 태스크 세밀 튜닝 + 운영 파이프라인 |

## 오늘(2026-04-19) 할 일

- R3 루프 20:00까지 완주, 결과 기록.
- 본 로드맵 commit + push.
- 내일 세션은 S1 준비(코퍼스 소스 조사)로 시작.
