#!/usr/bin/env python
"""General-purpose LLMs vs dedicated guard models on the unified flag protocol.

Reuses build_leaderboard's tested metric functions. Reports guard FA/miss/ECE/AURC
per benchmark, the general-LLM baseline (mean over the 6 general models) on the same
benchmark, and---critically---splits IN-DISTRIBUTION cells (a guard evaluated on the
benchmark it was trained on: WildGuard->WildGuard, Aegis->Aegis) from OUT-OF-DISTRIBUTION
cells, so a guard's home-turf advantage is not mistaken for general reliability.
Ethics is excluded (invalid for content-safety guards).
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_leaderboard import cell_metrics, load_solo, MODELS

GUARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "results", "guard-results")
GUARDS = ["llamaguard3", "wildguard", "shieldgemma", "aegis"]
GBENCH = ["xstest", "wildguard", "aegis", "toxicchat", "beavertails"]
INDIST = {("wildguard", "wildguard"), ("aegis", "aegis")}  # guard trained on that data


def guard_load(guard, ds):
    p = os.path.join(GUARD_DIR, f"{guard}__{ds}.jsonl")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    out = {}
    for line in open(p):
        r = json.loads(line)
        s = r["solo"]
        if not s.get("json_parsed", True) or s.get("flag_prob") is None:
            continue
        gt = 1 if r["gt"] == "(A)" else 0
        pred = 1 if s["answer_lp"] == "(A)" else 0
        out[str(r["id"])] = dict(gt=gt, pred=pred, conf=float(s["conf_lp"]),
                                 flag_prob=float(s["flag_prob"]), gen_conf=None)
    return out or None


def scalars(m):
    return dict(n=int(m["n"]), acc=float(m["acc"]), macro_f1=float(m["macro_f1"]),
                fa=float(m["fa"]), miss=float(m["miss"]), ece=float(m["ece"]),
                aurc=float(m["aurc"]))


print("=== GUARD cells (IN = trained on this benchmark) ===")
print(f"{'guard':12} {'benchmark':11} {'dist':4} {'n':>4} {'FA':>6} {'miss':>6} {'ECE':>6} {'AURC':>6}")
guard_rows = {}
for g in GUARDS:
    for ds in GBENCH:
        d = guard_load(g, ds)
        if not d:
            continue
        m = scalars(cell_metrics(list(d.values())))
        guard_rows[(g, ds)] = m
        tag = "IN " if (g, ds) in INDIST else "OUT"
        print(f"{g:12} {ds:11} {tag:4} {m['n']:>4} {m['fa']:>6.3f} {m['miss']:>6.3f} "
              f"{m['ece']:>6.3f} {m['aurc']:>6.3f}")

print("\n=== GENERAL-LLM baseline (mean over the 6 general models) ===")
print(f"{'benchmark':11} {'FA':>6} {'miss':>6} {'ECE':>6} {'AURC':>6}")
gen_rows = {}
for ds in GBENCH:
    ms = [cell_metrics(list(load_solo(mm, ds).values())) for mm in MODELS if load_solo(mm, ds)]
    if not ms:
        continue
    gen_rows[ds] = {k: float(np.mean([x[k] for x in ms]))
                    for k in ("fa", "miss", "ece", "aurc")}
    r = gen_rows[ds]
    print(f"{ds:11} {r['fa']:>6.3f} {r['miss']:>6.3f} {r['ece']:>6.3f} {r['aurc']:>6.3f}")

# Guard vs general summary: mean ECE/AURC across IN vs OUT cells
print("\n=== guard vs general: mean ECE / AURC ===")
for label, cells in [("guard IN-dist", [(g, ds) for (g, ds) in guard_rows if (g, ds) in INDIST]),
                     ("guard OUT-dist", [(g, ds) for (g, ds) in guard_rows if (g, ds) not in INDIST])]:
    if not cells:
        continue
    ece = np.mean([guard_rows[c]["ece"] for c in cells])
    aurc = np.mean([guard_rows[c]["aurc"] for c in cells])
    print(f"{label:16} cells={len(cells):2}  ECE={ece:.3f}  AURC={aurc:.3f}")
gb = [ds for ds in GBENCH if ds in gen_rows]
if gb:
    print(f"{'general (all)':16} bench={len(gb):2}  ECE={np.mean([gen_rows[d]['ece'] for d in gb]):.3f}  "
          f"AURC={np.mean([gen_rows[d]['aurc'] for d in gb]):.3f}")

json.dump({f"{g}|{ds}": v for (g, ds), v in guard_rows.items()}
          | {f"general|{ds}": v for ds, v in gen_rows.items()},
          open(os.path.join(HERE, "guard_results.json"), "w"), indent=2)
print("\nwrote guard_results.json  (guard cells:", len(guard_rows), ")")
