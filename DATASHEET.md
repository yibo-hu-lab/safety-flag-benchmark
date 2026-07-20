# Datasheet for Safety-Flag

This datasheet follows the structure of *Datasheets for Datasets* (Gebru et al., 2021).
Safety-Flag releases **derived artifacts only** (item IDs, harmonized labels, the standardized
prompt, and per-item model outputs); it redistributes no source-benchmark text.

## Motivation

**For what purpose was the dataset created?**
To evaluate LLM content moderators on *operational reliability* rather than accuracy alone. It
recasts six widely used safety benchmarks into one balanced binary flag / do-not-flag protocol and
records, per item, each model's verdict and two confidence signals, so a moderator can be audited on
three separable axes: error direction (over- vs. under-flagging), probability calibration, and
selective ranking (coverage–risk abstention).

**Who created the dataset and who funded it?**
Yibo Hu (Illinois Institute of Technology). See the paper and `CITATION.cff`.

## Composition

**What do the instances represent?**
Each instance is one *model decision on one content item*: the item's source ID and source benchmark,
the harmonized gold flag label, the model's parsed verdict, its token-logprob confidence, and its
verbalized (1–10, rescaled to [0,1]) confidence. Instances are grouped per model × benchmark in
`data/results/` as JSONL. **No prompt or benchmark text is included.**

**How many instances are there?**
The evaluation suite is ≈1,200 items (198–200 class-balanced items per benchmark × 6 benchmarks;
each benchmark's item set is the intersection of IDs present for every model). Every item is scored
by 10 moderators (6 general-purpose LLMs + 4 dedicated guards) plus 3 reference models
(gpt-4.1-mini, gpt-5.4-mini, R1-Distill-Llama-8B), yielding the per-item output files. Additional
cells cover prompt-robustness (paraphrase / label-swap) and a mixed-precision Qwen2.5-32B scale point.

**Is there a label, and how was it derived?**
Yes: a binary **flag / do-not-flag** label. The label is **not a fresh annotation.** Each item keeps
its source benchmark's own released label, mapped deterministically to flag / do-not-flag by
a fixed rule per benchmark (BeaverTails: any harm category present; XSTest: unsafe-contrast prompt;
Ethics: immoral action; WildGuard: harmful prompt; Aegis: unsafe interaction; ToxiChat: toxic
content). The full mapping is in the paper (Table 1) and the released loaders.

**Is any information missing?**
gpt-5.4-mini is a reasoning model whose API hides option-token logprobs, so its logprob-based ECE
and AURC are absent (verdict and verbalized confidence are present). Two of ≈7,600 items failed
JSON parsing (<0.03%).

**Are there errors, noise, or redundancies?**
Label noise is inherited from the sources. It is highest on Aegis (see label validation below).

## Collection process

**How was the data associated with each instance acquired?**
Model outputs were produced by running each moderator through one shared inference harness on the
identical item set, under one standardized flag-plus-confidence prompt (see `README.md`). Verdicts
are decoded greedily. Guards are scored through their native safety interfaces and mapped to the flag
decision. Source item IDs and native labels come from the six public benchmarks.

**Over what timeframe was the data collected?**
Model outputs were generated in 2026 for the accompanying paper. Model identifiers are recorded in
the per-item files and the paper's setup section.

## Preprocessing / cleaning / labeling

**Was any preprocessing done?**
Each benchmark was subsampled to a class-balanced ≈200-item set (fixed seed) and its native label
mapped to the binary flag label by the fixed rule above. The evaluation set per benchmark is the
intersection of item IDs present for every model.

**How was the label harmonization validated?**
By **two independent LLM judges from different vendors** — `gpt-5.5` and `claude-opus-4-8`, neither
among the evaluated systems. Each was shown only the item content under the released flag policy,
blind to the mapped gold, and re-labeled the same 150-item balanced sample (25 per benchmark). Both
agree with the mapped gold (gpt-5.5 Cohen's κ = 0.89; Claude κ = 0.87) and agree with each other more
strongly still (judge-vs-judge κ = 0.92). Disagreement concentrates on Aegis (κ = 0.68 and 0.61),
whose source labels are noisier; Aegis cells are read with more caution. Details are in the paper.

## Uses

**What (other) tasks could the dataset be used for?**
1. **Model selection.** Compare candidate moderators on error direction, calibration, and triage
   quality on identical items, instead of a single accuracy number.
2. **Reliability research.** The released per-item verdicts and confidences are a ready testbed for
   calibration, selective-prediction / abstention, and human-escalation methods, with no model rerun.
Because the item set is fixed, results stay comparable as new moderators are added.

**Is there anything that should NOT be done with the dataset?**
The labels encode the policy of each source benchmark and this suite's harmonization; they are not a
universal definition of "unsafe." Report per-benchmark results; do not treat one mapping as ground
truth for all moderation. The suite is English, single-turn, binary, and ≈200 items per benchmark —
sufficient for the reliability reads in the paper, not a production-scale audit. It reports aggregate
error only (no group- or dialect-conditional breakdown).

## Distribution

**How is the dataset distributed, and under what license?**
Openly on GitHub (`https://github.com/yibo-hu-lab/safety-flag-benchmark`) and archived on Zenodo
under the **concept DOI [10.5281/zenodo.21429763](https://doi.org/10.5281/zenodo.21429763)**, which
resolves to the latest versioned snapshot. Code (`analysis/`, `reproduce.sh`) is MIT; derived data
(`data/`, labels, prompt, outputs) is CC-BY-4.0. Source-benchmark text is **not** redistributed and
remains under each source's own license.

**Provenance.** Each item's source benchmark is recorded via the `dataset` field. Raw text must be
obtained from the original source:

| Benchmark | Source | Source license |
|---|---|---|
| BeaverTails | PKU-Alignment/BeaverTails | CC BY-NC 4.0 |
| XSTest | paul-rottger/xstest | CC BY 4.0 |
| Ethics | hendrycks/ethics | MIT |
| WildGuard | allenai/wildguardmix | ODC-BY (gated) |
| Aegis | nvidia/Aegis-AI-Content-Safety-Dataset-2.0 | CC BY 4.0 |
| ToxiChat | Baheti et al., 2021 | see source |

## Maintenance

**Who maintains the dataset, and how can it be contacted?**
Yibo Hu (yhu89@illinoistech.edu). Issues and requests via the GitHub repository.

**Will the dataset be updated, and how are versions handled?**
Versioned snapshots are archived on Zenodo; the concept DOI above always resolves to the latest.
Corrections and added models are released as new versions with updated `CITATION.cff` and changelog.

**How can others contribute or extend it?**
New moderators can be scored on the released item IDs and prompt and added to the leaderboard via the
`analysis/` scripts; `reproduce.sh` regenerates every table and figure from the per-item outputs with
no GPU or model download.
