# 선행연구 조사 — Legal Lane + Z3 솔버 소형 LLM

**작성:** 연구원 1 (sonnet) / 2026-04-18
**범위:** 법률 추론 + 형식 솔버, LegalBench 태스크, Qwen 소형 모델, 법률 도메인 SFT

---

## a. 법률 추론 + 형식 솔버(Z3/SAT/SMT) 기존 연구

### SatLM — Satisfiability-Aided Language Models (NeurIPS 2023)
- LLM으로 자연어 문제를 선언적 사양으로 파싱한 뒤 SMT 솔버(Z3)에 위임. LSAT, BoardgameQA, StructuredRegex SOTA, PAL 대비 GSM 서브셋에서 +23%.
- **시사점:** LSAT(법률 유사 추론)에서 솔버 효과를 이미 증명 — 본 연구가 LegalBench로 확장하면 차별화 확보 가능. 다만 소형 모델(1~3B) 실험은 **미확인**.
- 출처: https://arxiv.org/abs/2305.09656 / https://github.com/xiye17/sat-lm

### Logic-LM (EMNLP Findings 2023)
- NL → 기호 사양 → 결정적 솔버(Z3/Prolog/Pyke) → self-refine. 평균 CoT 대비 +18.4%, 프롬프트 대비 +39.2%. ProofWriter/PrOntoQA/FOLIO/LogicalDeduction/AR-LSAT.
- **시사점:** L04 논리 판단 + self-refine 구조를 거의 그대로 채택 가능. 법률 태스크 실험은 AR-LSAT에 한정 — LegalBench 적용은 공백.
- 출처: https://arxiv.org/abs/2305.12295 / https://github.com/teacherpeterpan/Logic-LLM

### NL2Logic — AST-Guided FOL Translation (2026)
- AST 중간 표현을 통해 NL → FOL → Z3/SMT-LIB 결정적 컴파일. 0.5~27B 모델 12종 평가, 구문 정확도 거의 100%, 의미 정확도 GCD 대비 +30%, Logic-LM 다운스트림 +31%.
- **시사점:** 소형 모델(0.5B 포함)에서도 solver grounding이 작동한다는 직접 근거. 본 연구의 tool_call 스키마 설계에 AST 패턴 참고.
- 출처: https://arxiv.org/html/2602.13237

### Legal2LogicICL (2026)
- 법률 케이스를 PROLEG 형식 논리 사실로 변환하는 in-context learning. Few-shot 다양성으로 일반화 개선.
- **시사점:** 법률 NL → 형식 표현 매핑이 존재 — Prolog/PROLEG 경로는 본 연구(Z3)와 직교. 소형 모델 수치 **미확인**.
- 출처: https://arxiv.org/html/2604.11699

### NL2FOL (ICLR/OpenReview 2024)
- NL → FOL 문장 단위 번역, 논리적 오류(fallacy) 탐지에 적용.
- **시사점:** 형식화 실패 시 fallback 전략 설계에 참고.
- 출처: https://arxiv.org/abs/2405.02318

**공백(차별화 포인트):** SatLM/Logic-LM은 LSAT/추론 벤치만, NL2Logic은 NLI만 다룸. "LegalBench 4개 태스크 + 1.5B 모델 + Z3 + Lane 분류 SFT" 조합은 **미확인**.

---

## b. LegalBench 태스크별 특성 및 Z3 적합성

LegalBench 전체: 162 태스크, 6 범주(issue-spotting, rule-recall, rule-application, rule-conclusion, interpretation, rhetorical-understanding). 대부분 balanced accuracy, 일부 F1.

### hearsay (Rule-application)
- 입력: 증거 서술. 출력: hearsay 여부(binary). 연방증거규칙 801(c) 요건(out-of-court statement, offered for truth) 매칭 구조.
- **Z3 적합성:** **높음**. 요건 2~3개 Boolean AND — L01 요건 매칭의 교과서적 케이스. 단, 미묘한 예외(non-assertive conduct 등) 때문에 요건 분해 NL 단계가 병목.
- 출처: https://hazyresearch.stanford.edu/legalbench/tasks/

### rule_qa (Rule-recall)
- 입력: 법 규칙에 대한 질문. 출력: 자유 텍스트 답. open-ended QA.
- **Z3 적합성:** **낮음**. 규칙을 "기억"하는 태스크 — 형식화 대상 자체가 없음. L06 직접 생성 또는 RAG가 적합.
- **재배치 권고:** plan.md의 L02 매핑을 재검토. rule_qa는 L06으로 이동 추천.
- 출처: https://hazyresearch.stanford.edu/legalbench/tasks/rule_qa.html

### personal_jurisdiction (Rule-application)
- 입력: 50개 manual 시나리오. 출력: 관할권 성립 여부. Minimum contacts + purposeful availment + fair play 다요건 테스트.
- **Z3 적합성:** **높음**. 다요건 AND/OR 구조 — L01 요건 매칭 + L02 규칙 적용 결합. 단 샘플 50개로 작아 평가 노이즈 큼.
- 출처: https://hazyresearch.stanford.edu/legalbench/tasks/

### textualism_tool (Rhetorical-understanding)
- 입력: 판례 발췌. 출력: 텍스트주의 도구 사용 여부(binary per tool, 태스크 2개).
- **Z3 적합성:** **낮음~중간**. 판사의 수사(rhetoric) 식별 — 해석/분류 태스크에 가까움. 형식화 이득 제한적.
- **재배치 권고:** L03 포섭보다 L06/L07(직접 분류)이 적합. Lane 5개 축소 시 제외 후보.
- 출처: https://hazyresearch.stanford.edu/legalbench/tasks/

**정리:** 4개 태스크 중 **hearsay, personal_jurisdiction**이 Z3 수혜 가능. rule_qa와 textualism_tool은 직접 분류/생성이 합리적.

---

## c. Qwen2.5-1.5B-Instruct / Qwen3-1.7B 법률 벤치마크

### Qwen2.5-1.5B-Instruct
- 공식 LegalBench 수치 **미확인**. Qwen2.5 블로그/기술문서는 GSM8K/MATH/MMLU 중심. Edge-size 타깃 명시.
- 출처: https://qwenlm.github.io/blog/qwen2.5-llm/

### Qwen3-1.7B
- Qwen3 Technical Report(arXiv 2505.09388): DeepSeek-R1-Distill-Qwen-1.5B/Llama-8B와 thinking 모드 비교. Qwen3-1.7B-Base가 Qwen2.5-3B-Base와 동등. LegalBench 수치 **미확인**.
- **시사점:** C1(thinking) baseline이 DeepSeek-R1-Distill-8B 수준일 수 있어 A3 목표치 설정에 주의.
- 출처: https://arxiv.org/abs/2505.09388 / https://huggingface.co/Qwen/Qwen3-1.7B

**결론:** 1.5~1.7B급에서 LegalBench 공개 수치 부재 → 본 연구의 A1/B1 baseline 측정이 **신규 기여**.

---

## d. 법률 도메인 SFT 최신 연구

### SaulLM-7B (arXiv 2024)
- Mistral-7B 기반, 30B 법률 토큰 continued pretraining + instruction tuning. LegalBench-Instruct 개선.
- **시사점:** 본 연구(1.5B + Lane)와 규모/접근 모두 대비군. SFT 레시피 참고.
- 출처: https://arxiv.org/abs/2403.03883

### SaulLM-54B & 141B (NeurIPS 2024)
- Mixtral 기반, 540B 법률 토큰 pretraining + 도메인 DPO. LegalBench-Instruct에서 GPT-4 상회(141B), Llama3/Mixtral 상회(54B).
- **시사점:** 파라미터 scaling이 정답인 반대 가설 — 본 연구는 "소형 + 솔버"로 도전.
- 출처: https://arxiv.org/html/2407.19584

### LEGAL-BERT (EMNLP Findings 2020)
- BERT-base를 법률 코퍼스로 사전학습. 분류/NER 중심, 생성형 추론 불가.
- **시사점:** 인코더 모델 — 본 연구와 직접 비교 대상 아님. 임베딩 활용 여지.
- 출처: https://arxiv.org/abs/2010.02559

### LawGPT (중국어 법률)
- 대규모 중국어 법률 문서 pretraining + 법률 SFT. 법률 상담/시험 타깃.
- **시사점:** 한국어 KLAC 데이터 활용 시 다국어 접근 레퍼런스.
- 출처: https://www.nature.com/articles/s41599-025-05924-3

### LegiLM (2024, NLLP Workshop)
- 데이터 컴플라이언스 특화 법률 LM fine-tune. 좁은 도메인 SFT 성공 사례.
- 출처: https://arxiv.org/html/2409.13721v1

**정리:** 법률 특화는 주로 7B+ continued pretraining 중심. 1.5B + Lane + 외부 솔버 조합은 **공백**.

---

## e. LegalBench HuggingFace 경로 / 평가 코드

- **HuggingFace 데이터셋:** `nguha/legalbench` — https://huggingface.co/datasets/nguha/legalbench
  - Downloads 1.7M, CC-BY-4.0, text-classification + question-answering, 영어, 162 태스크 CSV.
- **공식 GitHub (코드/프롬프트/평가):** HazyResearch/legalbench — https://github.com/HazyResearch/legalbench/
  - 평가: 대다수 balanced-accuracy, 일부 F1. `UsingLegalBench.ipynb`에 로딩/평가 예시. 태스크별 폴더에 prompt template 포함.
- **프로젝트 홈페이지:** https://hazyresearch.stanford.edu/legalbench/
- **논문:** arXiv 2308.11462.
- **시사점:** 평가 스크립트를 그대로 이식 가능. 프롬프트 템플릿은 baseline A1/B1에서 그대로 사용 권장.

---

## f. jihye-moon/klac_legal_aid_counseling 데이터셋

- **경로:** https://huggingface.co/datasets/jihye-moon/klac_legal_aid_counseling
- **출처:** 대한법률구조공단(KLAC) 법률구조상담 웹페이지 크롤링.
- **규모:** 1K~10K (실제 Size category 태그 기준), CSV, 한국어.
- **태스크 카테고리:** conversational, text-classification (HF 태그).
- **필드 구성 세부(질문/답변/법령 라벨 등):** README에 상세 스키마 **미확인** — 실제 로딩 후 연구원 2가 확인 필요.
- **시사점:** L06 한국어 설명 SFT + L09 법률 용어 참고. 규모가 작아 증강 필수. 라이선스/상업 이용 조건 **미확인** — 연구용 사용 전 확인.

---

## 종합 시사점

1. **기존 연구 커버리지:** 솔버+LLM은 LSAT/NLI까지. LegalBench + 소형(1.5B) + Lane 조합은 공백 — 차별화 유효.
2. **Lane 재배치:** rule_qa(L02→L06), textualism_tool(L03→L06/L07) 재매핑 권장. hearsay/personal_jurisdiction은 Z3 수혜 태스크.
3. **비교 기준선:** 1.5~1.7B LegalBench 공개 수치 부재 → A1/B1 측정 자체가 기여.
4. **경쟁 모델:** SaulLM-54B/141B가 상한선. 목표는 "절대 1위"가 아니라 "파라미터 대비 효율" 입증.
5. **불확실 항목:** Qwen 소형 모델 LegalBench 수치, KLAC 데이터셋 세부 스키마, SatLM/Logic-LM의 1~3B 실험 — 모두 **미확인**, 추가 조사는 Director 지시 시에만.
