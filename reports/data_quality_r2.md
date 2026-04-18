# R2 Data Quality Report

**Generated:** 2026-04-19 07:59

---

## 1. SFT Dataset (`data/sft_r2.jsonl`)

**Total samples:** 9,000

### 1.1 Language Distribution

| Language | Count | % |
|----------|-------|---|
| English | 6,977 | 77.5% |
| Korean  | 2,023 | 22.5% |

### 1.2 Lane Distribution

| Lane | Count | Target |
|------|-------|--------|
| L01 | 4,000 | 4,000 |
| L05 | 1,000 | 1,000 |
| L06 | 3,000 | 3,000 |
| L09 | 500 | 500 |
| L10 | 500 | 500 |

### 1.3 Final Answer Length

| Metric | Value |
|--------|-------|
| Average | 639 chars |
| Minimum | 156 chars |
| Maximum | 1223 chars |
| Below 30 chars (discard threshold) | 0 |

### 1.4 Schema Validation

| Metric | Value |
|--------|-------|
| Tool-using samples | 5,000 |
| Schema pass | 5,000 |
| Schema pass rate | 100.0% |
| Roundtrip pass rate | 100.0% |

---

## 2. DPO Dataset (`data/dpo_r2.jsonl`)

**Total pairs:** 10,000

### 2.1 Rejected Sample Types

| Type | Count | % |
|------|-------|---|
| bad_matching | 3,000 | 30.0% |
| bad_method | 5,000 | 50.0% |
| short_answer | 2,000 | 20.0% |

---

## 3. Outlines Smoke Test

**Status:** PASSED
**Backend:** outlines

Schema compilation tests passed:
- `L01_TOOL_SCHEMA` → `outlines.json_schema()` OK
- `L05_TOOL_SCHEMA` → `outlines.json_schema()` OK
- `LANE_CHOICE_PATTERN` → `outlines.regex()` OK

---

## 4. R2 Design Conformance

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| Schema pass rate | ≥85% | 100.0% | PASS |
| Roundtrip pass rate | ≥85% | 100.0% | PASS |
| English ratio | 70% | 77.5% | PASS |
| Avg final answer length | ≥80 chars | 639 | PASS |
| Samples below 30 chars | 0 | 0 | PASS |
| Lane count | 5 | 5 | PASS |
| Allowed methods only | 100% | 100.0% | PASS |

---

## 5. Conclusion

R2 data pipeline completed. Key improvements over R1:
- **Lane reduced to 5** (L01, L05, L06, L09, L10): eliminates L02/L03/L04 classification noise
- **English 70%+**: addresses R1 language mismatch root cause
- **Final answer avg >600 chars**: far above R1's 11.5-char average
- **Schema pass rate 100%**: all tool_calls validated against R2 strict schema
- **Outlines integrated**: constrained decoding blocks method hallucination
- **DPO includes bad-method rejected samples**: explicit hallucination suppression signal
