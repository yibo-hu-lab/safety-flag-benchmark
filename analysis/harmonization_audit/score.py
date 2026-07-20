#!/usr/bin/env python3
"""Reproduce the label-harmonization audit: Cohen's kappa of each independent
judge against the mapped gold, and the judge-vs-judge kappa.

Reads (all content-free; keyed by source item id):
  item_meta.json    [{uid, benchmark}]
  gold.json         uid -> "(A)" | "(B)"     (our mapped gold: source label -> flag)
  judge_gpt5.json   uid -> 0 | 1              (judge 1: gpt-5.5;         1 = flag = (A))
  judge_claude.json uid -> 0 | 1              (judge 2: claude-opus-4-8; 1 = flag = (A))

No source-benchmark text is needed or shipped. Run: python score.py
"""
import json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
meta = json.load(open(os.path.join(HERE, "item_meta.json")))
gold = json.load(open(os.path.join(HERE, "gold.json")))
gpt = json.load(open(os.path.join(HERE, "judge_gpt5.json")))
cla = json.load(open(os.path.join(HERE, "judge_claude.json")))
bench_of = {m["uid"]: m["benchmark"] for m in meta}
gold01 = {u: (1 if v == "(A)" else 0) for u, v in gold.items()}

ORDER = ["beavertails", "xstest", "ethics", "wildguard", "aegis", "toxicchat"]
DISP = {"beavertails": "BeaverTails", "xstest": "XSTest", "ethics": "Ethics",
        "wildguard": "WildGuard", "aegis": "Aegis", "toxicchat": "ToxiChat"}


def kappa(a, b):
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(a) / n; pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, ((po - pe) / (1 - pe) if (1 - pe) > 0 else float("nan"))


def report(title, A, B):
    print(f"\n=== {title} ===")
    by = collections.defaultdict(lambda: {"a": [], "b": []})
    allA, allB = [], []
    for u in A:
        if u not in B or u not in bench_of:
            continue
        by[bench_of[u]]["a"].append(A[u]); by[bench_of[u]]["b"].append(B[u])
        allA.append(A[u]); allB.append(B[u])
    print(f"{'benchmark':<13}{'n':>4}{'agree%':>8}{'kappa':>8}")
    for bm in ORDER:
        a, b = by[bm]["a"], by[bm]["b"]
        if not a:
            continue
        po, k = kappa(a, b)
        print(f"{DISP[bm]:<13}{len(a):>4}{po*100:>7.1f}{k:>8.3f}")
    po, k = kappa(allA, allB)
    print(f"{'OVERALL':<13}{len(allA):>4}{po*100:>7.1f}{k:>8.3f}")


report("gpt-5.5 vs gold", gpt, gold01)
report("claude-opus-4-8 vs gold", cla, gold01)
report("gpt-5.5 vs claude-opus-4-8 (judge-vs-judge)", gpt, cla)
