# Label-harmonization audit

Evidence that the deterministic map from each source benchmark's own label to the binary
**flag / do-not-flag** decision is coherent and consistently interpretable under one written policy.

Two independent LLM judges from different vendors — **gpt-5.5** and **claude-opus-4-8**, neither among
the evaluated systems — re-labeled the same balanced 150-item sample (25 per benchmark), shown only
the item content under the released policy (`judge_prompt.txt`) and blind to the mapped gold.

| | vs. gold (overall Cohen's κ) | Aegis κ |
|---|---|---|
| gpt-5.5 | 0.89 | 0.68 |
| claude-opus-4-8 | 0.87 | 0.61 |
| judge-vs-judge | **0.92** | 0.91 |

The two judges agree with each other (κ = 0.92) more than either agrees with gold; disagreement
concentrates on Aegis, whose source labels are noisier.

## Files (content-free)

- `judge_prompt.txt` — the released flag policy shown to each judge.
- `item_meta.json` — `[{uid, benchmark}]` for the 150-item sample.
- `gold.json` — `uid -> "(A)"|"(B)"`, our mapped gold (each source's own label mapped to flag).
- `judge_gpt5.json`, `judge_claude.json` — `uid -> 0|1` (1 = flag), one file per judge.
- `score.py` — recomputes every κ above from the files here.

```bash
python score.py
```

## Reproducing the judge labels

No source-benchmark **text** is redistributed here. The item content shown to the judges is
reconstructed from the six source benchmarks via the item IDs in `item_meta.json` (see the top-level
`README.md` for sources and licenses). Given that content and the released `judge_prompt.txt`, each
judge is queried once per item (temperature 0) and its `A`/`B` verdict recorded as `1`/`0`.
