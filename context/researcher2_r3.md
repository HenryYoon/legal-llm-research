# Context for 연구원 2 — Round 3

**From:** Director
**Date:** 2026-04-19 12:10 KST
**Based on:** results/analysis_r2.md
**기한:** 20:00 KST HARD STOP (약 7.5시간 남음)

---

## 1. R2 실패 요약

- R2 A3 mean 0.402 = R1 A3와 동일. B1 0.412 미달, A1 0.454 미달.
- Trace 분석 결과 근본 원인 3가지:
  1. **Pipeline 파싱 버그**: `scripts/eval_with_solver_r2.py`의 `run_lane_solver_r2()`가 Outlines Lane 선택 후 생성된 텍스트에서 tool_call이 `final_answer`에 문자열로 누출. `lane`은 L01인데 기록은 L06으로 찍힘.
  2. **영문 SFT 커버리지 부족**: hearsay/PJ 영문 입력에서 tool_call 발화 시 `element_of_purpose`, `element_of_malice` 같은 generic 영어 단어 환각.
  3. **Binary 답변 포맷 부재**: Yes/No 태스크에서 paragraph 출력 → parse_rate 28-35%.

## 2. R3 지시 — 우선순위 순

### P0-A: Pipeline 파싱 버그 수정 (최우선)

`scripts/eval_with_solver_r2.py`의 `run_lane_solver_r2()` 및 `evaluate_task()` 검사:

1. Outlines로 lane 선택 → 그 lane에 따라 조건부 생성이 제대로 분기되는지 확인.
2. 만약 lane=L01이면 Outlines `json_schema(L01_TOOL_SCHEMA)`로 tool_call 강제 생성. 현재는 free-generation이 섞일 가능성.
3. Final answer 생성 전에 **tool_call 실행 결과가 실제 assistant 컨텍스트에 주입**되는지 확인.
4. Fallback: final_answer 안에 `<tool_call>...</tool_call>` 문자열이 있으면 정규식으로 추출 + 재실행 + 결과 기반 final answer 재생성.

**완료 조건**: 수정 후 hearsay 10개 샘플 smoke test에서 tool_call_rate ≥ 0.7, parse_rate ≥ 0.7.

### P0-B: 영문 SFT 시드 대량 확충

현재 영문 L01 시드가 소수. R3에서는 LegalBench 공식 태스크의 **법리 요건을 그대로 시드 요건명에 반영**.

**hearsay 200개** — 요건 화이트리스트:
- `out_of_court_statement` (발언이 법정 외에서 이뤄졌는가)
- `offered_for_truth` (주장된 사실의 진실 입증을 위한 것인가)
- 두 요건 모두 참이면 hearsay (보통 "No" = 증거능력 없음이라는 의미로 평가되나 태스크 라벨 규칙 확인)

**PJ 200개** — 요건:
- `minimum_contacts` (피고가 주 내에서 최소한의 연결성을 가졌는가)
- `purposeful_availment` (주 내 활동 이득을 의도적으로 활용했는가)
- `fair_play_and_substantial_justice` (관할권 행사가 공정성에 부합하는가)
- 세 요건 모두 참이면 PJ 성립.

**textualism 200개** — 요건:
- `dictionary_definition_used`
- `plain_meaning_invoked`
- 둘 중 하나 이상 참이면 textualism_tool.

**형식 예시 (영어):**
```json
{
  "messages": [
    {"role": "system", "content": "You are a legal reasoning AI. Classify the input into a Legal Lane and invoke the appropriate solver."},
    {"role": "user", "content": "On the issue of whether James is smart, the fact that a witness testified out of court that James graduated at the top of his class."},
    {"role": "assistant", "content": "<lane>L01</lane>\n<tool_call>{\"tool\":\"z3\",\"method\":\"check_elements\",\"elements\":[\"out_of_court_statement\",\"offered_for_truth\"],\"facts\":[\"witness testified outside court\",\"used to prove James is smart\"],\"matching\":{\"out_of_court_statement\":true,\"offered_for_truth\":true},\"mode\":\"and\"}</tool_call>"},
    {"role": "tool", "content": "{\"all_met\": true}"},
    {"role": "assistant", "content": "Yes. This is hearsay under FRE 801(c) because the statement was made out of court and offered to prove the truth of the matter asserted."}
  ]
}
```

**비중**: 영문 80% / 한국어 20%로 조정.

### P1-A: Binary 응답 강제

Outlines decoder에서 태스크 타입이 binary면 final_answer도 `^(Yes|No)\\b.*` regex constrained generation 적용. `scripts/eval_with_solver_r2.py`의 decoder.generate_free() 호출부 수정.

### P1-B: Schema 화이트리스트 엄격화

`solver/schemas_r2.py`의 L01 `elements` 필드를 `enum`으로 정의:
- hearsay, PJ, textualism 등 태스크별 요건 이름만 허용
- enum 밖 이름은 Outlines 단에서 차단

### P2: Reasoning 블록 (L06만, 시간 남으면)

L06 샘플에 `<reasoning>` 블록 추가. 이번 라운드 시간이 부족하면 skip.

## 3. 출력 산출물

- `scripts/eval_with_solver_r2.py` — pipeline 버그 fix
- `solver/schemas_r2.py` — elements enum 추가
- `data/seeds_r3/` — 영문 600+ 시드 (hearsay 200 + PJ 200 + textualism 200 + 기타)
- `data/sft_r3.jsonl` — 8~10K 샘플 (영문 80%)
- `scripts/generate_data_r3.py`
- `reports/data_quality_r3.md`
- `configs/sft_qwen25_r3.yaml`

## 4. 완료 보고

5~8줄, 포함 항목:
- Pipeline 버그 fix 여부 + smoke test 결과 (tool_call_rate, parse_rate)
- 영문 시드 개수 및 schema enum 적용 여부
- SFT 데이터 총 샘플, 언어 분포, final answer 길이
- R3-3 SFT 진행 준비 여부
- 예상 소요 시간 (20:00 HARD STOP 고려)

## 5. 시간 제약

- 현재 12:10 KST, 20:00까지 **7.5시간**.
- 데이터 생성 1~2시간, SFT 학습 1~1.5시간, 평가 1시간 예상 → **R3는 4시간 내 완료 필수**.
- 만약 영문 시드 600개 생성이 3시간 이상 걸릴 것 같으면 각 태스크 100개로 축소해도 됨. 품질 > 수량.
