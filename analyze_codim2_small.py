"""Does noise select a definite split in codimension 2, and is it the deterministic one?

Criteria (fixed before looking at the results):
  definite     the spread between noise seeds (mean pairwise L1 between their occupancy vectors)
               shrinks as the step size falls;
  on the set   the distance of the seed-averaged occupancy to the Filippov feasible set goes to 0;
  different    at the smallest steps the L1 gap between the seed-averaged stochastic occupancy and the
               Euler (deterministic) occupancy stays clearly above the seed spread and does not trend to 0.

    python analyze_codim2_small.py
"""

import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from filippov.dynamics import instance
from filippov.extensions import detect_chattering, distance_to_feasible, ode_traj, pattern_drifts

HERE = os.path.dirname(os.path.abspath(__file__))
L1 = lambda p, q: float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def main():
    small = {(r["seed"], r["w"]): r for r in json.load(open(os.path.join(HERE, "results", "ext_codim2_small.json")))}
    base = {(r["seed"], r["w"]): r for r in json.load(open(os.path.join(HERE, "results", "ext_codim2.json"))) if r["feasible"]}
    steps = sorted({float(a) for r in small.values() for a in r["occ"]}, reverse=True)
    per = {a: {"spread": [], "gap_ode": [], "dist_set": []} for a in steps}
    per_case = []
    for key, r in small.items():
        b = base[key]
        inst = instance(r["seed"])
        gaps, thetas = ode_traj(inst, r["w"])
        _, theta, _ = detect_chattering(gaps, thetas)
        H, N = pattern_drifts(inst, theta, r["w"], r["states"])
        row = {"seed": r["seed"], "w": r["w"], "width": max(hi - lo for lo, hi in b["feasible_ranges"])}
        for a in steps:
            occ = np.array(r["occ"][str(a)])
            spread = np.mean([L1(x, y) for x, y in itertools.combinations(occ, 2)])
            mean = occ.mean(0)
            per[a]["spread"].append(spread)
            per[a]["gap_ode"].append(L1(mean, b["ode_occ_fine"]))
            per[a]["dist_set"].append(distance_to_feasible(H, N, mean))
            row[str(a)] = {"spread": spread, "gap_ode": L1(mean, b["ode_occ_fine"])}
        per_case.append(row)
    lines = ["| step size | seed spread (median L1) | distance to Filippov set (median) | gap to deterministic occupancy (median L1) | cases where gap > 2 × spread |",
             "|---|---|---|---|---|"]
    for a in steps:
        sp, gp = np.array(per[a]["spread"]), np.array(per[a]["gap_ode"])
        lines.append(f"| {a:g} | {np.median(sp):.3f} | {np.median(per[a]['dist_set']):.3f} | {np.median(gp):.3f} | "
                     f"{np.mean(gp > 2 * sp):.0%} |")
    width = np.median([c["width"] for c in per_case])
    lines += ["", f"{len(per_case)} cases; median width of the Filippov feasible set {width:.3f}."]
    md = "\n".join(lines) + "\n"
    open(os.path.join(HERE, "results", "codim2_small.md"), "w").write(md)
    json.dump(per_case, open(os.path.join(HERE, "results", "codim2_small_cases.json"), "w"))
    print(md)

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for k, label, col in [("spread", "spread between noise seeds", "tab:blue"),
                          ("gap_ode", "gap to deterministic (Euler) occupancy", "tab:red"),
                          ("dist_set", "distance to Filippov feasible set", "tab:green")]:
        ax.loglog(steps, [np.median(per[a][k]) for a in steps], "o-", color=col, label=label)
    ax.axhline(width, color="gray", ls=":", label="width of feasible set")
    ax.set_xlabel("step size a"); ax.set_ylabel("median L1"); ax.legend(fontsize=7)
    ax.set_title("codimension 2: what the stochastic algorithm selects", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "codim2_selection.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
