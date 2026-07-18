#!/usr/bin/env python3
"""Frontier API anchor rows for tab_leaderboard (reproducibility).

Computes the leaderboard row for each closed API anchor (gpt-4.1-mini,
gpt-5.4-mini) on the SAME common item-set as the open models, using the paper's
canonical metrics: acc/F1/FA/miss from build_leaderboard.cell_metrics, and
*pooled* equal-width ECE + pooled AURC (matching the open-model table column,
which is pooled, not the per-cell macro-average). Reasoning models that expose no
option-token logprobs (logprob_ok < 50% of items) get ECE/AURC = None (reported
as --- in the table); their verbalized-confidence pooled ECE is reported instead.

Anchor jsonls live in backups/asr/results/anchor/ (produced by openai_anchor.py
with --ids-from locking identical item ids). Run: python anchor_leaderboard.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_leaderboard as BL
from rigor_analysis import ece_equalwidth

ANCHOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "results", "anchor")
REASONING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "results", "reasoning")
# Each reference-row prefix maps to the directory holding its per-item jsonls.
SOURCES = {"GPT-4.1-mini": ANCHOR_DIR, "gpt-5.4-mini": ANCHOR_DIR,
           "R1-Distill-Llama-8B": REASONING_DIR}
ANCHORS = list(SOURCES)


def load_anchor(prefix, ds):
    fn = f"{prefix}__{ds}.jsonl" if ds in ("beavertails", "xstest", "ethics") \
        else f"{prefix}__{ds}_s0.jsonl"
    p = os.path.join(SOURCES[prefix], fn)
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        r = json.loads(line); s = r["solo"]
        out[str(r["id"])] = dict(
            gt=1 if r["gt"] == "(A)" else 0,
            pred=1 if s["answer_lp"] == "(A)" else 0,
            conf=float(s["conf_lp"]),
            gen_conf=(float(s["gen_conf"]) / 10.0 if s.get("gen_conf") is not None else None),
            lp_ok=bool(s.get("logprob_ok")))
    return out


def common_items(prefix, ds):
    a = load_anchor(prefix, ds)
    if a is None:
        return None
    common = set(a.keys())
    for m in BL.MODELS:
        r = BL.load_solo(m, ds)
        if r:
            common &= set(r.keys())
    return [a[i] for i in sorted(common)]


rows = {}
for prefix in ANCHORS:
    per_cell = {k: [] for k in ("acc", "macro_f1", "fa", "miss")}
    pooled, pooled_corr, pooled_gc, pooled_gc_corr, lp = [], [], [], [], []
    for d in BL.DATASETS:
        items = common_items(prefix, d)
        if not items:
            continue
        mt = BL.cell_metrics(items)
        for k in per_cell:
            per_cell[k].append(mt[k])
        for v in items:
            corr = 1.0 if v["gt"] == v["pred"] else 0.0
            pooled.append(v["conf"]); pooled_corr.append(corr); lp.append(v["lp_ok"])
            if v["gen_conf"] is not None:
                pooled_gc.append(v["gen_conf"]); pooled_gc_corr.append(corr)
    lp_frac = float(np.mean(lp)) if lp else 0.0
    # A model has usable logprob confidence when conf_lp actually varies (real
    # option-token logprobs), not when a per-item flag says so: R1-Distill has
    # genuine logprobs whose flag was left unset, while gpt-5.4-mini exposes none
    # (constant conf_lp). Key on the confidence variance.
    has_lp = len(pooled) > 0 and float(np.std(np.array(pooled))) > 1e-4
    ece = float(ece_equalwidth(np.array(pooled), np.array(pooled_corr))) if has_lp else None
    aurc = float(BL.coverage_risk(np.array(pooled), np.array(pooled_corr))[0]) if has_lp else None
    ece_vb = float(ece_equalwidth(np.array(pooled_gc), np.array(pooled_gc_corr))) if pooled_gc else None
    rows[prefix] = dict(
        acc=float(np.nanmean(per_cell["acc"])), f1=float(np.nanmean(per_cell["macro_f1"])),
        fa=float(np.nanmean(per_cell["fa"])), miss=float(np.nanmean(per_cell["miss"])),
        ece=ece, aurc=aurc, ece_verb=ece_vb, logprob_ok_frac=lp_frac)

json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "anchor_results.json"), "w"), indent=2)
print(f"{'model':14} {'acc':>6} {'F1':>6} {'FA':>6} {'miss':>6} {'ECE':>7} {'AURC':>7} {'ECEv':>6}")
for m, r in rows.items():
    ece = f"{r['ece']:.3f}" if r['ece'] is not None else "N/A"
    aurc = f"{r['aurc']:.3f}" if r['aurc'] is not None else "N/A"
    ecev = f"{r['ece_verb']:.3f}" if r['ece_verb'] is not None else "N/A"
    print(f"{m:14} {r['acc']:6.3f} {r['f1']:6.3f} {r['fa']:6.3f} {r['miss']:6.3f} "
          f"{ece:>7} {aurc:>7} {ecev:>6}")
print("\nwrote anchor_results.json")
