"""Tables (results/extensions.md) and figures for the extension experiments (run_extensions.py).

    python analyze_extensions.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
STEPS = ["0.008", "0.004", "0.002", "0.001", "0.0005"]
L1 = lambda p, q: float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def stepsize(md):
    rows = json.load(open(os.path.join(RES, "ext_stepsize.json")))
    a = np.array([r["alpha"] for r in rows])
    md += ["### Step-size limit (codimension 1)", "",
           f"{len(rows)} sliding cases re-run with the ODE time horizon fixed (600 time units).", "",
           "| step size | median \\|occupancy − α\\| | mean | correlation with α | churn (flips per update, median) |",
           "|---|---|---|---|---|"]
    med, push = [], np.array([r["push"] for r in rows])
    groups = {"strong push (> 0.02)": push > 0.02, "weak push (< 0.005)": push < 0.005}
    by_group = {k: [] for k in groups}
    for s in STEPS:
        o = np.array([r["by_step"][s]["occupancy"] for r in rows])
        c = np.array([r["by_step"][s]["churn"] for r in rows])
        e = abs(o - a)
        med.append(np.median(e))
        for k, m in groups.items():
            by_group[k].append(np.median(e[m]))
        md.append(f"| {s} | {np.median(e):.3f} | {e.mean():.3f} | {np.corrcoef(a, o)[0, 1]:.2f} | {np.median(c):.4f} |")
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    x = [float(s) for s in STEPS]
    ax.loglog(x, med, "ko-", label="all cases")
    for (k, v), col in zip(by_group.items(), ["tab:green", "tab:red"]):
        ax.loglog(x, v, "o--", color=col, label=k)
    ax.loglog(x, med[0] * np.sqrt(np.array(x) / x[0]), ":", color="gray", label="∝ √a")
    ax.set_xlabel("step size a"); ax.set_ylabel("median |occupancy − α|"); ax.legend(fontsize=8)
    ax.set_title("stochastic occupancy converges to the Filippov weight", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "stepsize_limit.png"), dpi=160); plt.close(fig)
    return md + [""]


def moving(md):
    rows = json.load(open(os.path.join(RES, "ext_moving.json")))
    md += ["### Moving target (Polyak averaging at rate β, step size 0.002)", "",
           "| β | cases | still churning (> 0.005 flips per update) | churn (median) | late windows with mixed greedy action (median) |",
           "|---|---|---|---|---|"]
    for b in (0.0, 1e-5, 1e-4):
        r = [x for x in rows if x["beta"] == b]
        md.append(f"| {b:g} | {len(r)} | {np.mean([x['churn'] > 0.005 for x in r]):.0%} | "
                  f"{np.median([x['churn'] for x in r]):.4f} | {np.median([x['late_mixed_windows'] for x in r]):.2f} |")
    return md + [""]


def codim2(md):
    rows = [r for r in json.load(open(os.path.join(RES, "ext_codim2.json"))) if r["feasible"]]
    width = [max(hi - lo for lo, hi in r["feasible_ranges"]) for r in rows]
    pr = [r for r in rows if r["product"]]
    md += ["### Codimension 2", "",
           f"{len(rows)} instances with two states chattering at once and a non-empty Filippov feasible set "
           f"(scan of 1,500 instances at each of ω = 1.0 and 1.3). Occupancies are over the four sign patterns; "
           f"distances are L1.", "",
           "| quantity | median |", "|---|---|",
           f"| width of the feasible set (largest range of any pattern's weight) | {np.median(width):.3f} |",
           f"| distance of the Euler occupancy to the feasible set, dt = 0.01 | {np.median([r['dist_ode'] for r in rows]):.4f} |",
           f"| same, dt = 0.0025 | {np.median([r['dist_ode_fine'] for r in rows]):.4f} |",
           f"| change in Euler occupancy between dt = 0.01 and 0.0025 | {np.median([L1(r['ode_occ'], r['ode_occ_fine']) for r in rows]):.4f} |",
           f"| distance of the stochastic occupancy to the feasible set, a = 0.002 | {np.median([r['dist_sa']['0.002'] for r in rows]):.3f} |",
           f"| same, a = 0.0005 | {np.median([r['dist_sa']['0.0005'] for r in rows]):.3f} |",
           f"| stochastic (a = 0.002) vs Euler (dt = 0.0025) occupancy | {np.median([L1(r['sa_occ']['0.002'], r['ode_occ_fine']) for r in rows]):.3f} |",
           f"| stochastic (a = 0.0005) vs Euler (dt = 0.0025) occupancy | {np.median([L1(r['sa_occ']['0.0005'], r['ode_occ_fine']) for r in rows]):.3f} |",
           f"| product of codimension-1 weights vs Euler occupancy ({len(pr)} cases with both weights defined) | {np.median([L1(r['product'], r['ode_occ_fine']) for r in pr]):.3f} |",
           ""]
    return md


def main():
    md = stepsize([])
    md = moving(md)
    md = codim2(md)
    open(os.path.join(RES, "extensions.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
