#!/usr/bin/env bash
# Reproduce every table and figure from the bundled per-item outputs.
# No GPU or model download required.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p tables figures
cd analysis
echo "[1/6] leaderboard + landscape ..."; python3 build_leaderboard.py
echo "[2/6] calibration / recalibration / abstention (rigor) ..."; python3 rigor_analysis.py
echo "[3/6] general-vs-guard ..."; python3 guard_compare.py
echo "[4/6] prompt robustness ..."; python3 prompt_robust_analysis.py
echo "[5/6] frontier + reasoning reference rows ..."; python3 anchor_leaderboard.py
echo "[6/6] prevalence + tables/figures ..."; python3 prevalence.py; python3 make_tables_figs.py
echo "DONE. Tables and figures written under analysis/."
