"""Stage 3: evaluate detection.
 - threshold sweep: flag as hallucination if conflict count >= t   (t=1 is the 'any conflict' rule)
 - K ablation: 'any conflict' rule using only the first k samples
 - optional manual label check:  python analyze.py --export-manual 40  ->  fill column label_correct (1/0)"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from common import read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default="data/detected.jsonl")
ap.add_argument("--export-manual", type=int, default=0)
args = ap.parse_args()

rows = read_jsonl(args.inp)
if not rows:
    raise SystemExit("No detected.jsonl yet. Run detect.py first.")
Path("results").mkdir(exist_ok=True)

if args.export_manual:
    pd.DataFrame([{"id": r["id"], "claim": r["claim"], "true_label_factual": r["label_true"],
                   "reference": r["reference"], "auto_label_hallucinated": r["hallucinated"],
                   "label_correct": ""} for r in rows[:args.export_manual]]).to_csv("data/manual_check.csv", index=False)
    print("wrote data/manual_check.csv - fill label_correct with 1 (auto label right) or 0 (wrong), then rerun analyze.py")
    raise SystemExit

y = np.array([r["hallucinated"] for r in rows])
C = np.array([r["conflicts"] for r in rows])
K = C.shape[1]
score = C.sum(axis=1)
print(f"N={len(y)}  hallucinated={int(y.sum())}  K={K}")


def metrics(pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"acc": accuracy_score(y, pred), "precision": p, "recall": r, "f1": f, "flagged_pct": 100 * pred.mean()}


auroc = roc_auc_score(y, score) if len(set(y)) == 2 else float("nan")
thr = pd.DataFrame([{"threshold": t, **metrics((score >= t).astype(int))} for t in range(1, K + 1)])
kab = pd.DataFrame([{"k": k, **metrics((C[:, :k].sum(axis=1) >= 1).astype(int))} for k in range(1, K + 1)])
thr.to_csv("results/threshold_table.csv", index=False)
kab.to_csv("results/k_ablation.csv", index=False)
print(f"\nAUROC of conflict count: {auroc:.3f}")
print("\nThreshold sweep (t=1 is the 'any conflict' rule):\n", thr.round(3).to_string(index=False))
print("\nK ablation ('any conflict' with first k samples):\n", kab.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(thr.threshold, thr.acc, marker="o", label="accuracy")
ax[0].plot(thr.threshold, thr.f1, marker="o", label="F1")
ax[0].set(xlabel="conflict-count threshold", title=f"Threshold sweep (AUROC={auroc:.2f})")
ax[0].legend()
ax[1].plot(kab.k, kab.acc, marker="o", label="accuracy")
ax[1].plot(kab.k, kab.f1, marker="o", label="F1")
ax[1].set(xlabel="K sampled references", title="Effect of K")
ax[1].legend()
for a in ax:
    a.axhline(0.5, color="gray", ls="--", lw=1, label="chance accuracy (0.5)")
    a.axhline(2/3, color="red", ls=":", lw=1, label="flag-everything F1 (0.67)")
    a.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/detection_plots.png", dpi=150)

mc = Path("data/manual_check.csv")
if mc.exists():
    df = pd.read_csv(mc)
    df = df[df["label_correct"].isin([0, 1])]
    if len(df):
        print(f"\nManual label check: {df['label_correct'].mean() * 100:.1f}% of {len(df)} auto labels judged correct")
