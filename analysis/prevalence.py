#!/usr/bin/env python3
"""
Prevalence- and cost-sensitive operating analysis (CPU, uses existing predictions).
Answers the reviewer point that ECE/rates on a balanced 50/50 benchmark do not
characterize a production deployment with low harmful prevalence and asymmetric costs.

Per-item cost of a policy at harmful prevalence pi and miss:false-alarm cost ratio lambda:
    R(pi, lambda) = (1-pi) * FA + pi * lambda * Miss      (C_FA=1, C_Miss=lambda)
We use each model's macro FA/Miss and report which flagger is cost-optimal in each regime.
"""
import json, os
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(OUT, "results.json")))
agg = R["model_agg"]
MODELS = ["Qwen2.5-32B", "gemma-2-9b", "Qwen2.5-7B", "Mistral-7B", "OLMo-2-7B", "Llama-3.1-8B"]

PIS = [0.01, 0.05, 0.10, 0.25, 0.50]
LAMS = [1, 5, 10]

def risk(m, pi, lam):
    fa = agg[m]["fa"]; miss = agg[m]["miss"]
    return (1 - pi) * fa + pi * lam * miss

# best model per (pi, lambda)
print("Best (lowest-cost) flagger per deployment regime:")
print(f"{'lambda':>6} " + " ".join(f"pi={p:>4}" for p in PIS))
best_grid = {}
for lam in LAMS:
    row = []
    for pi in PIS:
        rs = {m: risk(m, pi, lam) for m in MODELS}
        best = min(rs, key=rs.get)
        best_grid[(lam, pi)] = best
        row.append(best)
    print(f"{lam:>6} " + " ".join(f"{b:>7}" for b in row))

# does the winner change vs the balanced equal-cost point (pi=.5, lam=1)?
base = best_grid[(1, 0.5)]
print(f"\nBalanced equal-cost winner (pi=.5, lam=1): {base}")
changed = sorted({b for b in best_grid.values() if b != base})
print("Other regimes elect different winners:", changed if changed else "none")

# accuracy-order vs deployment-order at a realistic low-prevalence, high-miss-cost regime
pi, lam = 0.05, 10
order = sorted(MODELS, key=lambda m: risk(m, pi, lam))
acc_order = sorted(MODELS, key=lambda m: -agg[m]["macro_f1"])
print(f"\nAt pi={pi}, lambda={lam} (low prevalence, misses 10x costlier):")
print("  cost order :", order)
print("  F1 order   :", acc_order)

json.dump({"best_grid": {f"{k[0]}|{k[1]}": v for k, v in best_grid.items()},
           "pis": PIS, "lams": LAMS}, open(os.path.join(OUT, "prevalence_results.json"), "w"), indent=2)

# ---- emit tex ----
PAPER = os.path.dirname(OUT)
DISP = {"Qwen2.5-32B": "32B", "gemma-2-9b": "Gemma", "Qwen2.5-7B": "7B",
        "Mistral-7B": "Mistral", "OLMo-2-7B": "OLMo", "Llama-3.1-8B": "Llama"}
with open(os.path.join(PAPER, "tables", "tab_prevalence.tex"), "w") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{\textbf{The cost-optimal flagger depends on the deployment regime, not the balanced
leaderboard.} Lowest-expected-cost model at harmful prevalence $\pi$ and miss-to-false-alarm cost ratio
$\lambda$, from $R=(1-\pi)\,\mathrm{FA}+\pi\lambda\,\mathrm{Miss}$ on each model's macro rates. The
balanced equal-cost cell ($\pi{=}0.5,\lambda{=}1$) elects the leaderboard winner (Qwen2.5-32B), but
where harmful content is rare and misses are not heavily penalized, the low-false-alarm OLMo-2-7B---next
to last on the balanced leaderboard---is cheapest. Balanced-benchmark rank does not by itself pick the
right production model.}
\label{tab:prevalence}
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{@{}l""" + "c" * len(PIS) + r"""@{}}
\toprule
\textbf{$\lambda$} & """ + " & ".join(r"$\pi{=}%g\%%$" % (p * 100) for p in PIS) + r""" \\
\midrule
""")
    for lam in LAMS:
        cells = " & ".join(DISP[best_grid[(lam, pi)]] for pi in PIS)
        f.write(f"{lam} & {cells} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
print("wrote tab_prevalence.tex")
