# Contributing

## 커밋 메시지 규약 — Conventional Commits (경량)

### 형식

```
<type>(<scope>): <subject 한국어/영어 혼용>

<body (선택) — WHY 중심, 72자 wrap>

Co-Authored-By: ...
```

### Type (7개만 사용)

| Type | 용도 |
|------|------|
| `feat` | 새 기능/데이터/실험 아티팩트 추가 (예: 새 솔버, 새 데이터셋, 새 평가 스크립트) |
| `fix` | 버그/결과 오류 수정 |
| `docs` | 문서/리포트/분석만 변경 (README, reports/, results/*.md 등) |
| `refactor` | 동작 변화 없이 구조만 변경 (파일 이동, 이름 변경) |
| `test` | 테스트 추가/수정 |
| `chore` | 의존성, 설정, 잡무 (initial commit 포함) |
| `perf` | 성능 개선 (속도/VRAM/토큰) |

### Scope (선택)

실험 라운드를 scope로 쓰는 것을 권장:
- `(r1)`, `(r2)`, `(r3)` — 라운드별 산출물
- `(solver)`, `(data)`, `(eval)`, `(train)` — 영역별

### 예시

```
feat(r2): Lane 축소, 영어 SFT, Outlines 통합
docs(r1): add A2/A3 results and final analysis
refactor: reorganize md files into docs/ and context/
fix(eval): correct parse_rate when <lane> prefix present
chore: bump torch to 2.4
```

### 규칙

1. subject는 50자 이내, 명령문("add", not "added").
2. 한국어/영어 혼용 OK. type은 영문 고정.
3. body가 있으면 subject 다음 빈 줄 하나 후 작성.
4. **WHY 중심**. WHAT은 diff로 알 수 있으니 "왜 그렇게 했는가" 위주.
5. Co-Authored-By는 Claude 작업분 표기용 footer.

### Subject에 안 쓰는 것

- 마침표(`.`) 끝에 붙이지 않음.
- "Update X" 같은 의미 없는 subject 지양 → `docs: update X with reason`처럼 이유 포함.

### 체크리스트

- [ ] type이 7개 중 하나인가?
- [ ] subject가 50자 이내인가?
- [ ] 명령문인가? (add, fix, remove, not added, fixed)
- [ ] body에 WHY가 있는가? (WHAT만 있으면 굳이 body 안 써도 됨)

---

## 브랜치 / 워크플로

- 라운드 단위 진행: R1, R2, R3 ...
- 각 라운드 안에서는 단계별 커밋 (context, 데이터, 학습, 평가, 분석).
- 가설 기각 시 `results/analysis.md` 업데이트 + `docs(rN): record failure analysis` 커밋.

## 예외

connventional commits의 `!` (breaking change) 표기나 changelog 자동 생성은 이 프로젝트에서 쓰지 않음. 필요해지면 도입.
