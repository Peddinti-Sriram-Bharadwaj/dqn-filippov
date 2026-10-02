"""Tables (results/summary.md) and figures (figures/*.png) from results/results.jsonl.

    python analyze.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from filippov import detect_sliding, filippov_alpha, instance, ode
from filippov.dynamics import A, GAMMA, S

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
WS = [0.7, 1.0, 1.3, 1.6]
COLORS = dict(zip(WS, ["tab:blue", "tab:green", "tab:orange", "tab:red"]))


def load():
    return [json.loads(l) for l in open(os.path.join(HERE, "results", "results.jsonl"))]


def summary(rows):
    out = ["| ω | instances with codimension-1 sliding | ODE occupancy − α (median abs.) | stochastic occupancy − α (median abs.) | correlation (α, stochastic occupancy) | within 0.05 | churn (flips per update, median) |",
           "|---|---|---|---|---|---|---|"]
    for w in WS + ["all"]:
        rw = rows if w == "all" else [r for r in rows if r["w"] == w]
        sl = [r for r in rw if r.get("attracting")]
        a = np.array([r["alpha"] for r in sl]); o = np.array([r["ode_occupancy"] for r in sl])
        e = np.array([r["sa_occupancy"] for r in sl]); c = np.array([r["sa_churn"] for r in sl])
        out.append(f"| {w} | {len(sl)} / {len(rw)} ({len(sl) / len(rw):.0%}) | {np.median(abs(a - o)):.3f} | "
                   f"{np.median(abs(a - e)):.3f} | {np.corrcoef(a, e)[0, 1]:.2f} | {np.mean(abs(a - e) < 0.05):.0%} | {np.median(c):.3f} |")
    sl = [r for r in rows if r.get("attracting")]
    push = np.array([min(abs(r["h_plus_n"]), abs(r["h_minus_n"])) for r in sl])
    err = np.array([abs(r["alpha"] - r["sa_occupancy"]) for r in sl])
    out += ["", "| weaker normal push min(‖h⁺ₙ‖, ‖h⁻ₙ‖) | cases | median abs. error (stochastic) |", "|---|---|---|"]
    for lo, hi in [(0, 0.005), (0.005, 0.02), (0.02, np.inf)]:
        m = (push >= lo) & (push < hi)
        out.append(f"| [{lo:g}, {hi:g}) | {m.sum()} | {np.median(err[m]):.3f} |")
    n_nonattracting = sum(1 for r in rows if r["sliding"] and not r.get("attracting"))
    out += ["", f"Detected chattering cases failing the Filippov attraction condition: {n_nonattracting}."]
    return "\n".join(out) + "\n"


def fig_scatter(rows):
    sl = [r for r in rows if r.get("attracting")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for w in WS:
        x = [r["alpha"] for r in sl if r["w"] == w]
        axes[0].scatter(x, [r["ode_occupancy"] for r in sl if r["w"] == w], s=14, color=COLORS[w], label=f"ω={w}")
        axes[1].scatter(x, [r["sa_occupancy"] for r in sl if r["w"] == w], s=14, color=COLORS[w], label=f"ω={w}")
    for ax, title in zip(axes, ["expected dynamics (Euler ODE)", "stochastic algorithm (constant step)"]):
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("Filippov weight α (predicted)"); ax.set_ylabel("fraction of time on the '+' side")
        ax.set_title(title, fontsize=10); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "occupancy_vs_alpha.png"), dpi=160); plt.close(fig)


def fig_error_vs_push(rows):
    sl = [r for r in rows if r.get("attracting")]
    fig, ax = plt.subplots(figsize=(5.6, 4))
    for w in WS:
        p = [min(abs(r["h_plus_n"]), abs(r["h_minus_n"])) for r in sl if r["w"] == w]
        e = [abs(r["alpha"] - r["sa_occupancy"]) for r in sl if r["w"] == w]
        ax.scatter(p, e, s=14, color=COLORS[w], label=f"ω={w}")
    ax.set_xscale("log"); ax.set_xlabel("weaker normal push  min(‖h⁺ₙ‖, ‖h⁻ₙ‖)")
    ax.set_ylabel("|stochastic occupancy − α|"); ax.legend(fontsize=8)
    ax.set_title("prediction error shrinks as the drift dominates the noise", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "error_vs_push.png"), dpi=160); plt.close(fig)


def fig_example(rows, seed=4, w=1.0):
    inst = instance(seed)
    gaps, thetas = ode(inst, w)
    s, theta = detect_sliding(gaps, thetas)
    alpha = filippov_alpha(inst, s, theta, w)[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    t = np.arange(len(gaps)) * 0.01
    axes[0].plot(t, gaps[:, s], color="k", lw=0.8)
    axes[0].axhline(0, color="tab:red", lw=1)
    axes[0].set_xlabel("ODE time"); axes[0].set_ylabel(f"Q(s{s}, 1) − Q(s{s}, 0)")
    axes[0].set_title(f"instance {seed}, ω={w}: the gap at s{s} is driven to 0 and held there", fontsize=9)
    late = gaps[-2000:, s]
    zoom = np.sign(late[-60:])
    axes[1].plot(np.arange(len(zoom)), zoom, drawstyle="steps-post", lw=1.2, color="tab:blue", marker=".")
    axes[1].set_ylim(-1.5, 1.5); axes[1].set_yticks([-1, 1]); axes[1].set_yticklabels(["action 0 greedy", "action 1 greedy"])
    axes[1].set_xlabel("last 60 Euler steps")
    axes[1].set_title(f"occupancy of '+' over 2,000 steps: {np.mean(late > 0):.3f}  (α = {alpha:.3f})", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "example_sliding.png"), dpi=160); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    rows = load()
    md = summary(rows)
    open(os.path.join(HERE, "results", "summary.md"), "w").write(md)
    print(md)
    fig_scatter(rows); fig_error_vs_push(rows); fig_example(rows)


if __name__ == "__main__":
    main()
