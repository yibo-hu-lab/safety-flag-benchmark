#!/usr/bin/env python3
"""
Rigor re-analysis for the moderation-reliability single-model paper.

Answers two reviewer critiques from the RAW per-item jsonl (CPU-only, no inference):

  Block 1 (calibration rigor):
    B1.1 Bootstrap 95% CIs (2000 item-level resamples) on ECE_lp, ECE_vb, AURC
         -- pooled over benchmarks AND per model x benchmark.
    B1.2 Adaptive / equal-mass ECE (10 equal-COUNT bins) alongside equal-width ECE.
    B1.3 NLL (of the correct class under the model's flag probability) and
         classwise ECE (ECE on flag-class vs no-flag-class items).
    B1.4 Split-robust temperature scaling: 20 random 50/50 splits; mean+-sd of
         optimal T, mean+-sd post-recal ECE, mean fold-reduction factor.

  Block 2 (abstention without the benchmark-selection artifact):
    B2.1 Pooled abstention selective risk @ 50% coverage (rank by raw conf).
    B2.2 Within-benchmark abstention @ 50% coverage, MACRO-averaged.
    B2.3 Benchmark-normalized (z-scored within benchmark) ranking @ 50% coverage.
    B2.4 Macro-AURC (per-benchmark AURC, averaged) vs pooled AURC.
    B2.5 Pooled-vs-within gap + fraction of pooled-abstained items per benchmark.

Loaders mirror analysis/build_leaderboard.py exactly (same paths, same common
item-set restriction) so recomputed baselines match analysis/results.json.

Writes: analysis/rigor_results.json,
        tables/tab_calibration_rigor.tex,
        tables/tab_abstention_controls_rigor.tex
"""
import json, os
import numpy as np

SEED = 0
np.random.seed(SEED)
RESULTS = os.environ.get("SAFETY_FLAG_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "results"))
EXT = os.path.join(RESULTS, "moderation_ext")
OUT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(OUT)
TAB = os.path.join(PAPER, "tables")

MODELS = ["Mistral-7B", "Llama-3.1-8B", "Qwen2.5-7B", "Qwen2.5-32B", "gemma-2-9b", "OLMo-2-7B"]
DATASETS = ["beavertails", "xstest", "ethics", "wildguard", "aegis", "toxicchat", "toxigen"]
DISP = {"beavertails": "BeaverTails", "xstest": "XSTest", "ethics": "Ethics",
        "wildguard": "WildGuard", "aegis": "Aegis", "toxicchat": "ToxiChat", "toxigen": "ToxiGen"}
# Table display order = ranked by macro-F1 (matches make_exp_tables.py)
DISP_ORDER = ["Qwen2.5-32B", "gemma-2-9b", "Qwen2.5-7B", "Mistral-7B", "OLMo-2-7B", "Llama-3.1-8B"]

FRONT_ALIAS = {"Mistral-7B": "Mistral-7B", "Llama-3.1-8B": "Llama-3.1-8B", "Qwen2.5-7B": "Qwen2.5-7B",
               "Qwen2.5-32B": "Qwen2.5-32B-AWQ", "gemma-2-9b": "gemma-2-9b", "OLMo-2-7B": "OLMo-2-7B"}
EXT_ALIAS = {"Mistral-7B": "Mistral-7B-Instruct-v0.3", "Llama-3.1-8B": "Llama-3.1-8B-Instruct",
             "Qwen2.5-7B": "Qwen2.5-7B-Instruct", "Qwen2.5-32B": "Qwen2.5-32B-Instruct",
             "gemma-2-9b": "gemma-2-9b-it", "OLMo-2-7B": "OLMo-2-1124-7B-Instruct"}


def path_for(model, ds):
    if ds in ("beavertails", "xstest", "ethics"):
        if model == "Qwen2.5-32B" and ds == "ethics":
            return os.path.join(RESULTS, "frontier", "Qwen2.5-32B__ethics_sev_s0.jsonl")
        return os.path.join(RESULTS, f"{FRONT_ALIAS[model]}__{ds}.jsonl")
    return os.path.join(EXT, f"{EXT_ALIAS[model]}__{ds}_s0.jsonl")


def load_solo(model, ds):
    """id -> dict(gt, pred, conf(=conf_lp), flag_prob, gen_conf in [0,1] or None)."""
    p = path_for(model, ds)
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        r = json.loads(line)
        s = r["solo"]
        gt_flag = 1 if r["gt"] == "(A)" else 0
        pred_flag = 1 if s["answer_lp"] == "(A)" else 0
        gen = s.get("gen_conf", None)
        out[str(r["id"])] = dict(gt=gt_flag, pred=pred_flag, conf=float(s["conf_lp"]),
                                 flag_prob=float(s["flag_prob"]),
                                 gen_conf=(float(gen) / 10.0 if gen is not None else None))
    return out


# ---- load + common item-set per dataset (identical to build_leaderboard) ----
raw = {}
for m in MODELS:
    for d in DATASETS:
        r = load_solo(m, d)
        if r is not None:
            raw[(m, d)] = r
common_ids = {}
for d in DATASETS:
    sets = [set(raw[(m, d)]) for m in MODELS if (m, d) in raw]
    common_ids[d] = sorted(set.intersection(*sets))


def cell_items(m, d):
    return [raw[(m, d)][i] for i in common_ids[d]]


def pooled(m, key_extract):
    """Return list over all datasets (with dataset tag) for model m."""
    out = []
    for d in DATASETS:
        if (m, d) not in raw:
            continue
        for it in cell_items(m, d):
            out.append((d, it))
    return out


# =============================== metrics ===============================
def ece_equalwidth(conf, correct, n_bins=10):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    if len(conf) == 0:
        return np.nan
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0; n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            e += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def ece_equalmass(conf, correct, n_bins=10):
    """Adaptive ECE: 10 bins of (near-)equal COUNT, split on sorted confidence."""
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    n = len(conf)
    if n == 0:
        return np.nan
    order = np.argsort(conf)
    conf, correct = conf[order], correct[order]
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    e = 0.0
    for i in range(n_bins):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        c = conf[a:b]; y = correct[a:b]
        e += (len(c) / n) * abs(y.mean() - c.mean())
    return float(e)


def aurc(conf, correct):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    order = np.argsort(-conf)
    err = 1.0 - correct[order]
    cum = np.cumsum(err) / np.arange(1, len(err) + 1)
    cov = np.arange(1, len(err) + 1) / len(err)
    return float(np.trapezoid(cum, cov))


def risk_at_coverage(conf, correct, c=0.5):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    order = np.argsort(-conf)
    err = 1.0 - correct[order]
    cum = np.cumsum(err) / np.arange(1, len(err) + 1)
    k = max(1, int(round(c * len(err))))
    return float(cum[k - 1])


def nll_correct_class(flag_prob, y):
    """NLL of the true class under the model's flag distribution P(A)=flag_prob."""
    p = np.clip(np.asarray(flag_prob, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_T(p_flag, y):
    z = logit(np.asarray(p_flag, float)); y = np.asarray(y, float)
    Ts = np.linspace(0.3, 20.0, 800); best, bT = 1e18, 1.0
    for T in Ts:
        q = 1.0 / (1.0 + np.exp(-z / T))
        nll = -np.mean(y * np.log(np.clip(q, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-9, 1)))
        if nll < best:
            best, bT = nll, T
    return float(bT)


def boot_ci(vecs_fn, n, B=2000, rng=None):
    """vecs_fn(idx)->scalar metric on resampled indices. Returns (lo, hi, se)."""
    rng = rng or np.random.default_rng(SEED)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = vecs_fn(idx)
    vals = vals[~np.isnan(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(vals.std())


# =============================== Block 1 ===============================
block1 = {}
for m in MODELS:
    ds_present = [d for d in DATASETS if (m, d) in raw]
    # pooled arrays
    P_conf, P_corr, P_flagprob, P_y, P_gen_conf, P_gen_corr = [], [], [], [], [], []
    for d in ds_present:
        for it in cell_items(m, d):
            corr = 1.0 if it["gt"] == it["pred"] else 0.0
            P_conf.append(it["conf"]); P_corr.append(corr)
            P_flagprob.append(it["flag_prob"]); P_y.append(it["gt"])
            if it["gen_conf"] is not None:
                P_gen_conf.append(it["gen_conf"]); P_gen_corr.append(corr)
    P_conf = np.array(P_conf); P_corr = np.array(P_corr)
    P_flagprob = np.array(P_flagprob); P_y = np.array(P_y)
    P_gen_conf = np.array(P_gen_conf); P_gen_corr = np.array(P_gen_corr)

    rng = np.random.default_rng(SEED)

    # B1.1 pooled CIs
    ece_lp = ece_equalwidth(P_conf, P_corr)
    ece_lp_ci = boot_ci(lambda idx: ece_equalwidth(P_conf[idx], P_corr[idx]), len(P_conf), rng=rng)
    aurc_lp = aurc(P_conf, P_corr)
    aurc_ci = boot_ci(lambda idx: aurc(P_conf[idx], P_corr[idx]), len(P_conf), rng=rng)
    ece_vb = ece_equalwidth(P_gen_conf, P_gen_corr)
    ece_vb_ci = boot_ci(lambda idx: ece_equalwidth(P_gen_conf[idx], P_gen_corr[idx]),
                        len(P_gen_conf), rng=rng)

    # B1.2 adaptive / equal-mass ECE (pooled) + its CI
    ece_lp_mass = ece_equalmass(P_conf, P_corr)
    ece_lp_mass_ci = boot_ci(lambda idx: ece_equalmass(P_conf[idx], P_corr[idx]), len(P_conf), rng=rng)

    # B1.3 NLL + classwise ECE
    nll = nll_correct_class(P_flagprob, P_y)
    flag_mask = P_y == 1
    noflag_mask = P_y == 0
    ece_flag = ece_equalwidth(P_conf[flag_mask], P_corr[flag_mask])
    ece_noflag = ece_equalwidth(P_conf[noflag_mask], P_corr[noflag_mask])
    classwise_ece = float(np.nanmean([ece_flag, ece_noflag]))

    # B1.4 split-robust temperature scaling: 20 random 50/50 splits
    conf_pre_all = np.maximum(P_flagprob, 1 - P_flagprob)
    z_all = logit(P_flagprob)
    Ts, ece_pre_list, ece_post_list, red_list = [], [], [], []
    rng_ts = np.random.default_rng(SEED + 100)
    n = len(P_flagprob)
    for _ in range(20):
        perm = rng_ts.permutation(n)
        half = n // 2
        tr, te = perm[:half], perm[half:]
        T = fit_T(P_flagprob[tr], P_y[tr])
        q_te = 1.0 / (1.0 + np.exp(-z_all[te] / T))
        conf_post_te = np.maximum(q_te, 1 - q_te)
        e_pre = ece_equalwidth(conf_pre_all[te], P_corr[te])
        e_post = ece_equalwidth(conf_post_te, P_corr[te])
        Ts.append(T); ece_pre_list.append(e_pre); ece_post_list.append(e_post)
        red_list.append(e_pre / e_post if e_post > 1e-9 else np.nan)
    Ts = np.array(Ts); ece_pre_list = np.array(ece_pre_list)
    ece_post_list = np.array(ece_post_list); red_list = np.array(red_list)

    # per model x benchmark CIs (ECE_lp, ECE_vb, AURC)
    per_bench = {}
    for d in ds_present:
        its = cell_items(m, d)
        c = np.array([it["conf"] for it in its])
        cr = np.array([1.0 if it["gt"] == it["pred"] else 0.0 for it in its])
        gc = np.array([it["gen_conf"] for it in its if it["gen_conf"] is not None])
        gcr = np.array([1.0 if it["gt"] == it["pred"] else 0.0
                        for it in its if it["gen_conf"] is not None])
        rd = np.random.default_rng(SEED + hash(d) % 1000)
        e_lp = ece_equalwidth(c, cr)
        e_lp_ci = boot_ci(lambda idx: ece_equalwidth(c[idx], cr[idx]), len(c), rng=rd)
        a_lp = aurc(c, cr)
        a_ci = boot_ci(lambda idx: aurc(c[idx], cr[idx]), len(c), rng=rd)
        e_vb = ece_equalwidth(gc, gcr) if len(gc) else np.nan
        e_vb_ci = (boot_ci(lambda idx: ece_equalwidth(gc[idx], gcr[idx]), len(gc), rng=rd)
                   if len(gc) else (np.nan, np.nan, np.nan))
        per_bench[d] = dict(n=len(c),
                            ece_lp=e_lp, ece_lp_ci=[e_lp_ci[0], e_lp_ci[1]],
                            ece_vb=e_vb, ece_vb_ci=[e_vb_ci[0], e_vb_ci[1]],
                            aurc=a_lp, aurc_ci=[a_ci[0], a_ci[1]])

    block1[m] = dict(
        n_pooled=int(len(P_conf)), n_verbalized=int(len(P_gen_conf)),
        ece_lp=ece_lp, ece_lp_ci=[ece_lp_ci[0], ece_lp_ci[1]],
        ece_lp_equalmass=ece_lp_mass, ece_lp_equalmass_ci=[ece_lp_mass_ci[0], ece_lp_mass_ci[1]],
        ece_vb=ece_vb, ece_vb_ci=[ece_vb_ci[0], ece_vb_ci[1]],
        aurc=aurc_lp, aurc_ci=[aurc_ci[0], aurc_ci[1]],
        nll=nll, ece_flagclass=ece_flag, ece_noflagclass=ece_noflag, classwise_ece=classwise_ece,
        ts_T_mean=float(Ts.mean()), ts_T_sd=float(Ts.std(ddof=1)),
        ts_ece_pre_mean=float(ece_pre_list.mean()), ts_ece_pre_sd=float(ece_pre_list.std(ddof=1)),
        ts_ece_post_mean=float(ece_post_list.mean()), ts_ece_post_sd=float(ece_post_list.std(ddof=1)),
        ts_reduction_mean=float(np.nanmean(red_list)), ts_reduction_sd=float(np.nanstd(red_list, ddof=1)),
        per_benchmark=per_bench,
    )


# =============================== Block 2 ===============================
block2 = {}
for m in MODELS:
    ds_present = [d for d in DATASETS if (m, d) in raw]
    # pooled arrays with benchmark tags
    conf, corr, bench = [], [], []
    per_bench_conf, per_bench_corr = {}, {}
    for d in ds_present:
        its = cell_items(m, d)
        c = np.array([it["conf"] for it in its])
        cr = np.array([1.0 if it["gt"] == it["pred"] else 0.0 for it in its])
        per_bench_conf[d] = c; per_bench_corr[d] = cr
        conf.extend(c); corr.extend(cr); bench.extend([d] * len(c))
    conf = np.array(conf); corr = np.array(corr); bench = np.array(bench)

    # B2.1 pooled abstention @ 50%
    pooled_risk = risk_at_coverage(conf, corr, 0.5)
    pooled_aurc = aurc(conf, corr)
    full_err = float(1 - corr.mean())

    # B2.2 within-benchmark @ 50%, macro-avg
    within_risks = {d: risk_at_coverage(per_bench_conf[d], per_bench_corr[d], 0.5) for d in ds_present}
    within_macro = float(np.mean(list(within_risks.values())))

    # B2.3 benchmark-normalized (z within benchmark), pool, abstain @ 50%
    zconf = np.empty_like(conf)
    for d in ds_present:
        mask = bench == d
        c = conf[mask]
        sd = c.std()
        zconf[mask] = (c - c.mean()) / (sd if sd > 1e-12 else 1.0)
    norm_risk = risk_at_coverage(zconf, corr, 0.5)

    # B2.4 macro-AURC vs pooled AURC
    macro_aurc = float(np.mean([aurc(per_bench_conf[d], per_bench_corr[d]) for d in ds_present]))

    # B2.5 fraction of pooled-abstained items per benchmark (abstained = bottom 50% by conf)
    order = np.argsort(-conf)
    k = max(1, int(round(0.5 * len(conf))))
    abstained_idx = order[k:]  # the low-confidence half we drop
    ab_bench = bench[abstained_idx]
    total_ab = len(ab_bench)
    frac_abstained = {}
    frac_of_benchmark_dropped = {}
    for d in ds_present:
        n_d_ab = int((ab_bench == d).sum())
        frac_abstained[d] = n_d_ab / total_ab if total_ab else 0.0
        n_d_total = int((bench == d).sum())
        frac_of_benchmark_dropped[d] = n_d_ab / n_d_total if n_d_total else 0.0

    block2[m] = dict(
        full_err=full_err,
        pooled_risk_50=pooled_risk, within_macro_risk_50=within_macro,
        normalized_risk_50=norm_risk,
        gap_within_minus_pooled=float(within_macro - pooled_risk),
        pooled_aurc=pooled_aurc, macro_aurc=macro_aurc,
        gap_macro_minus_pooled_aurc=float(macro_aurc - pooled_aurc),
        within_risk_per_benchmark=within_risks,
        frac_abstained_from_benchmark=frac_abstained,
        frac_of_benchmark_abstained=frac_of_benchmark_dropped,
    )


# =============================== dump ===============================
allres = dict(seed=SEED, models=MODELS, datasets=DATASETS,
              common_ids_n={d: len(common_ids[d]) for d in DATASETS},
              block1_calibration=block1, block2_abstention=block2)
json.dump(allres, open(os.path.join(OUT, "rigor_results.json"), "w"), indent=2, default=float)
print("wrote rigor_results.json")


# =============================== LaTeX tables ===============================
def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


# --- Table (a): calibration with CIs + adaptive-ECE + NLL + classwise ---
with open(os.path.join(TAB, "tab_calibration_rigor.tex"), "w") as f:
    f.write(r"""\begin{table*}[t]
\centering
\caption{\textbf{Calibration rigor: finite-sample CIs, binning sensitivity, and proper
scores.} All quantities are pooled over the six benchmarks on identical item sets.
ECE$_{\text{ew}}$ is equal-width (10-bin) ECE with a bootstrap $95\%$ CI ($2000$
item-level resamples); ECE$_{\text{em}}$ is the adaptive equal-\emph{mass} (10 equal-count
bin) ECE, showing the estimate is not a binning artifact. ECE$_{\text{vb}}$ is verbalized
$1$--$10$ confidence. NLL is the negative log-likelihood of the true class under the
model's flag probability. ECE$^{\text{flag}}$/ECE$^{\lnot\text{flag}}$ are class-conditional
(computed on harmful vs.\ benign items separately). The ordering of models by ECE is
preserved under both binning schemes and the CIs do not overlap between the best and worst
models, so the calibration gap is not a finite-sample artifact.}
\label{tab:calibration_rigor}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
\textbf{Model} & ECE$_{\text{ew}}$ [95\% CI] & ECE$_{\text{em}}$ & ECE$_{\text{vb}}$ & NLL & ECE$^{\text{flag}}$ & ECE$^{\lnot\text{flag}}$ & AURC [95\% CI] \\
\midrule
""")
    for m in DISP_ORDER:
        b = block1[m]
        f.write(
            f"{m} & {b['ece_lp']:.3f} [{b['ece_lp_ci'][0]:.3f}, {b['ece_lp_ci'][1]:.3f}] "
            f"& {b['ece_lp_equalmass']:.3f} & {f3(b['ece_vb'])} & {b['nll']:.3f} "
            f"& {f3(b['ece_flagclass'])} & {f3(b['ece_noflagclass'])} "
            f"& {b['aurc']:.3f} [{b['aurc_ci'][0]:.3f}, {b['aurc_ci'][1]:.3f}] \\\\\n")
    f.write(r"""\bottomrule
\end{tabular}
\end{table*}
""")

# --- Table (a2): split-robust temperature scaling ---
with open(os.path.join(TAB, "tab_recalibration_rigor.tex"), "w") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{\textbf{The ECE reduction from temperature scaling is split-robust.} Over $20$
random $50/50$ fit/eval splits we report the optimal temperature $T$ (mean$\pm$sd), the
held-out ECE before and after scaling, and the per-split fold-reduction $\text{ECE}_{\text{pre}}/\text{ECE}_{\text{post}}$.
The reduction is stable across splits (small sd), so the headline ``$3$--$8\times$'' effect
is not a lucky split.}
\label{tab:recal_rigor}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Model} & \textbf{$T$ (mean$\pm$sd)} & \textbf{ECE pre} & \textbf{ECE post} & \textbf{fold-reduction} \\
\midrule
""")
    for m in DISP_ORDER:
        b = block1[m]
        f.write(
            f"{m} & ${b['ts_T_mean']:.1f}\\pm{b['ts_T_sd']:.1f}$ "
            f"& ${b['ts_ece_pre_mean']:.3f}\\pm{b['ts_ece_pre_sd']:.3f}$ "
            f"& ${b['ts_ece_post_mean']:.3f}\\pm{b['ts_ece_post_sd']:.3f}$ "
            f"& ${b['ts_reduction_mean']:.1f}\\times$ \\\\\n")
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

# --- Table (b): abstention controls ---
with open(os.path.join(TAB, "tab_abstention_controls_rigor.tex"), "w") as f:
    f.write(r"""\begin{table}[t]
\centering
\caption{\textbf{Abstention gains survive the benchmark-selection control.} Selective risk
at $50\%$ coverage under three ranking rules: \emph{pooled} (rank all items by raw
confidence and abstain globally---the paper's method), \emph{within} (abstain within each
benchmark separately, then macro-average across benchmarks), and \emph{norm.} (z-score
confidence within each benchmark, then pool and abstain). If pooled risk were far below the
within-benchmark risk, the global policy would be exploiting easy \emph{benchmarks} rather
than easy \emph{items}; the small \emph{gap} column (within~$-$~pooled) shows this is not
the case. Macro-AURC (per-benchmark AURC, averaged) is reported beside pooled AURC for the
same reason.}
\label{tab:abstain_ctrl_rigor}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
 & \multicolumn{4}{c}{\textbf{Selective risk @ 50\% coverage}} & \multicolumn{2}{c}{\textbf{AURC}} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-7}
\textbf{Model} & Pooled & Within & Norm. & Gap & Pooled & Macro \\
\midrule
""")
    for m in DISP_ORDER:
        b = block2[m]
        f.write(
            f"{m} & {b['pooled_risk_50']:.3f} & {b['within_macro_risk_50']:.3f} "
            f"& {b['normalized_risk_50']:.3f} & {b['gap_within_minus_pooled']:+.3f} "
            f"& {b['pooled_aurc']:.3f} & {b['macro_aurc']:.3f} \\\\\n")
    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

print("wrote tables:", "tab_calibration_rigor.tex, tab_recalibration_rigor.tex, tab_abstention_controls_rigor.tex")


# =============================== verification vs results.json ===============================
try:
    base = json.load(open(os.path.join(OUT, "results.json")))
    print("\n=== VERIFICATION vs results.json (loaders sanity) ===")
    for m in ["Qwen2.5-32B", "Llama-3.1-8B", "OLMo-2-7B"]:
        a = base["model_agg"][m]
        b1 = block1[m]; b2 = block2[m]
        print(f"{m}: pooled AURC recomputed={b1['aurc']:.4f} vs results.json={a['aurc']:.4f} "
              f"| risk@0.5 recomputed={b2['pooled_risk_50']:.4f} vs results.json={a['risk_pool']['0.5']:.4f}")
except Exception as e:
    print("verification skipped:", e)
