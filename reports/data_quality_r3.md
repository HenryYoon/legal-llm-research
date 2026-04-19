# R3 Data Quality Report

**Overall: PASS**

## Checks

| Check | Value | Target | Status |
|-------|-------|--------|--------|
| Sample count | 8500 | ≥ 8000 | PASS |
| EN ratio | 87.5% | ≥ 70% | PASS |
| Binary Yes/No start | 1.000 (6390/6390) | ≥ 0.95 | PASS |
| Element whitelist violations | 0 | 0 | PASS |
| Tool_call JSON valid | 1.000 | ≥ 0.95 | PASS |
| Avg final_answer length | 190.4 chars | ≥ 60 | PASS |

## Lane Distribution

- L01: 7450 (87.6%)
- L05: 223 (2.6%)
- L06: 528 (6.2%)
- L09: 153 (1.8%)
- L10: 146 (1.7%)

## Source Distribution (top 15)

- legalbench_textualism_r3: 1505
- legalbench_hearsay_r3: 1501
- legalbench_pj_r3: 1497
- korean_civil_r3: 1060
- legalbench_textualism_test_r3: 813
- legalbench_hearsay_test_r3: 699
- legal_explanation_r3: 528
- legalbench_pj_test_r3: 375
- legal_calc_r3: 223
- legal_translation_r3: 153
- legal_uncertain_r3: 146

## Language Split

- EN: 7440 (87.5%)
- KO: 1060 (12.5%)