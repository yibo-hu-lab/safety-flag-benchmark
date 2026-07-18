#!/usr/bin/env python3
"""Emit LaTeX tables + PDF figures from results.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(OUT)
TAB = os.path.join(PAPER, "tables")
FIG = os.path.join(PAPER, "figures")
R = json.load(open(os.path.join(OUT, "results.json")))
cells, agg = R["cells"], R["model_agg"]

MODELS = ["Mistral-7B", "Llama-3.1-8B", "Qwen2.5-7B", "Qwen2.5-32B", "gemma-2-9b", "OLMo-2-7B"]
DATASETS = ["beavertails", "xstest", "ethics", "wildguard", "aegis", "toxicchat"]
DISP = {"beavertails": "BeaverTails", "xstest": "XSTest", "ethics": "Ethics",
        "wildguard": "WildGuard", "aegis": "Aegis", "toxicchat": "ToxiChat"}
def C(m, d): return cells.get(f"{m}|{d}")

order = sorted(MODELS, key=lambda m: -agg[m]["macro_f1"])

# ---------- Table 1: leaderboard ----------
with open(os.path.join(TAB, "tab_leaderboard.tex"), "w") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{\textbf{Reliability \& calibration leaderboard} for six open-weight LLMs as
binary content-moderation flaggers, macro-averaged over the six benchmarks
(identical item sets; \S\ref{sec:bench}). \emph{FA} = false-alarm rate (benign content
flagged; over-caution); \emph{Miss} = harmful content not flagged (under-caution);
\emph{ECE} on logprob confidence; \emph{AURC} = area under the coverage--risk curve
(lower is better, \S\ref{sec:abstain}). Rows ranked by macro-F1.}
\label{tab:leaderboard}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Model} & \textbf{Acc} & \textbf{F1} & \textbf{FA}$\downarrow$ & \textbf{Miss}$\downarrow$ & \textbf{ECE}$\downarrow$ & \textbf{AURC}$\downarrow$ \\
\midrule
""")
    for m in order:
        a = agg[m]
        f.write(f"{m} & {a['acc']:.3f} & {a['macro_f1']:.3f} & {a['fa']:.3f} & "
                f"{a['miss']:.3f} & {a['ece']:.3f} & {a['aurc']:.3f} \\\\\n")
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---------- Table 2: over-/under-caution landscape (FA / Miss per cell) ----------
with open(os.path.join(TAB, "tab_landscape.tex"), "w") as f:
    f.write(r"""\begin{table*}[t]
\centering
\caption{\textbf{The over-/under-caution landscape}: false-alarm~/~miss rate for every
model$\times$benchmark cell (identical items). A cell reads \textbf{FA\,/\,Miss}: the
left number is benign content wrongly flagged, the right is harmful content missed. The
two error modes are anti-correlated across models---Llama-3.1-8B flags almost everything
(FA up to $0.98$), OLMo-2-7B flags almost nothing (miss up to $0.90$)---so a single
accuracy number hides opposite failure profiles.}
\label{tab:landscape}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}l""" + "c" * len(DATASETS) + r"""@{}}
\toprule
\textbf{Model} & """ + " & ".join(r"\textbf{%s}" % DISP[d] for d in DATASETS) + r""" \\
\midrule
""")
    for m in order:
        row = [m]
        for d in DATASETS:
            c = C(m, d)
            row.append("--" if c is None else f"{c['fa']:.2f}\\,/\\,{c['miss']:.2f}")
        f.write(" & ".join(row) + r" \\" + "\n")
    f.write(r"""\bottomrule
\end{tabular}
\end{table*}
""")

# ---------- Table 3: calibration + abstention ----------
with open(os.path.join(TAB, "tab_calibration.tex"), "w") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{\textbf{Calibration and reliability recovered by abstention.} ECE$_{\text{lp}}$
uses token-logprob confidence; ECE$_{\text{vb}}$ uses the model's verbalized 1--10
confidence. \emph{Risk@$c$} is the error rate when the flagger answers only on its most
confident fraction $c$ of items and abstains on the rest (pooled over benchmarks). A large
drop from Risk@1.0 to Risk@0.5 means errors are \emph{knowable} from confidence; a flat
column (Llama, OLMo) means they are not.}
\label{tab:calibration}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
 & \multicolumn{3}{c}{\textbf{Calibration}} & \multicolumn{3}{c}{\textbf{Risk@coverage}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
\textbf{Model} & ECE$_{\text{lp}}$ & ECE$_{\text{vb}}$ & Brier & 1.0 & 0.8 & 0.5 \\
\midrule
""")
    for m in order:
        a = agg[m]
        rp = a["risk_pool"]
        f.write(f"{m} & {a['ece']:.3f} & {a['ece_verb']:.3f} & {a['brier']:.3f} & "
                f"{rp['1.0']:.3f} & {rp['0.8']:.3f} & {rp['0.5']:.3f} \\\\\n")
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---------- Figure 1: teaser scatter FA vs Miss ----------
plt.rcParams.update({"font.size": 11, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(4.2, 3.6))
colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
for m, col in zip(MODELS, colors):
    fa = [C(m, d)["fa"] for d in DATASETS if C(m, d)]
    ms = [C(m, d)["miss"] for d in DATASETS if C(m, d)]
    ax.scatter(fa, ms, color=col, s=28, alpha=0.55, edgecolors="none")
    ax.scatter(np.mean(fa), np.mean(ms), color=col, s=150, marker="*",
               edgecolors="k", linewidths=0.6, label=m, zorder=5)
ax.plot([0, 1], [0, 1], ls=":", c="gray", lw=0.8)
ax.set_xlabel("False-alarm rate  (over-caution)")
ax.set_ylabel("Miss rate  (under-caution)")
ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.02, 0.65)
ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
ax.set_title("Each model occupies a different error regime", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_landscape.pdf"))
plt.close(fig)

# ---------- Figure 2: coverage-risk curves ----------
fig, ax = plt.subplots(figsize=(4.2, 3.4))
for m, col in zip(order, plt.cm.tab10(np.linspace(0, 1, len(MODELS)))):
    xs = np.array([1.0, 0.8, 0.5])
    ys = np.array([agg[m]["risk_pool"]["1.0"], agg[m]["risk_pool"]["0.8"],
                   agg[m]["risk_pool"]["0.5"]])
    ax.plot(xs, ys, "-o", color=col, ms=4, lw=1.4, label=f"{m}")
ax.set_xlabel("Coverage (fraction answered)")
ax.set_ylabel("Selective risk (error on answered)")
ax.invert_xaxis()
ax.legend(fontsize=7, loc="upper left")
ax.set_title("Abstention helps calibrated models most", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_coverage_risk.pdf"))
plt.close(fig)

# ---------- Figure 3: reliability diagrams, well- vs poorly-calibrated ----------
import sys
sys.path.insert(0, OUT)
from build_leaderboard import load_solo, common_ids  # noqa: E402
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
for ax, m in zip(axes, ["Qwen2.5-32B", "Llama-3.1-8B"]):
    confs, corr = [], []
    for d in DATASETS:
        r = load_solo(m, d)
        if r is None: continue
        for i in common_ids[d]:
            v = r[i]
            confs.append(v["conf"]); corr.append(1.0 if v["gt"] == v["pred"] else 0.0)
    confs = np.array(confs); corr = np.array(corr)
    bins = np.linspace(0, 1, 11)
    xs, ys = [], []
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        msk = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if msk.sum() == 0: continue
        xs.append(confs[msk].mean()); ys.append(corr[msk].mean())
    ax.plot([0, 1], [0, 1], ls=":", c="gray", lw=0.9)
    ax.bar(np.array(xs), np.array(ys), width=0.08, color="#4C72B0", alpha=0.85, edgecolor="k", linewidth=0.4)
    ax.set_title(f"{m}\nECE$_{{lp}}$={agg[m]['ece']:.3f}", fontsize=9)
    ax.set_xlabel("Confidence"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
axes[0].set_ylabel("Accuracy")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_reliability.pdf"))
plt.close(fig)

print("wrote tables to", TAB)
print("wrote figures to", FIG)
print("leaderboard order:", order)
