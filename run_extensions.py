"""Experiments toward Borkar's (2026) new results and toward policy churn.

  stepsize   codimension-1 sliding cases re-run at step sizes 0.008 ... 0.0005 with the ODE time
             horizon fixed: does the stochastic occupancy converge to the Filippov weight alpha as the
             step shrinks, and how does churn scale?
  codim2     scan random instances for sliding on the intersection of two tie hyperplanes; there the
             Filippov tangency conditions leave a one-parameter family of possible splits between the
             four sign patterns. Record where the expected (Euler, two step sizes) and stochastic (two
             step sizes) dynamics land in that family, and a product-of-codimension-1 baseline.
  moving     codimension-1 cases with a slowly moving target (Polyak averaging at rate beta): does
             sliding, i.e. mixed greedy occupancy, persist as the manifold moves?

    python run_extensions.py stepsize codim2 moving --workers 3
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from filippov.dynamics import filippov_alpha, instance
from filippov.extensions import (detect_chattering, distance_to_feasible, filippov_feasible_set,
                                 ode_traj, pattern_drifts, stochastic_track)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
HORIZON = 600.0
STEPS = [0.008, 0.004, 0.002, 0.001, 0.0005]


def codim1_cases(n, seed=0):
    rows = [json.loads(l) for l in open(os.path.join(RES, "results.jsonl"))]
    sl = [r for r in rows if r.get("attracting")]
    rng = np.random.default_rng(seed)
    return [sl[i] for i in rng.choice(len(sl), size=min(n, len(sl)), replace=False)]


def _stepsize(r):
    inst = instance(r["seed"])
    out = {"seed": r["seed"], "w": r["w"], "state": r["state"], "alpha": r["alpha"],
           "push": min(abs(r["h_plus_n"]), abs(r["h_minus_n"])), "by_step": {}}
    for a in STEPS:
        t = stochastic_track(inst, r["w"], [r["state"]], a, HORIZON, seed=r["seed"])
        out["by_step"][str(a)] = {"occupancy": t["pattern_occupancy"][1], "churn": t["churn"][0]}
    return out


def _scan(args):
    seed, w = args
    inst = instance(seed)
    gaps, thetas = ode_traj(inst, w)
    states, theta, at_rest = detect_chattering(gaps, thetas)
    if len(states) != 2 or not at_rest:
        return None
    H, N = pattern_drifts(inst, theta, w, list(states))
    feas = filippov_feasible_set(H, N)
    if feas is None:
        return {"seed": seed, "w": w, "states": states.tolist(), "feasible": False}
    late = gaps[-len(gaps) // 4:][:, states]
    def occ_from_gaps(G):
        bits = (G > 0).astype(int)
        codes = bits[:, 0] * 2 + bits[:, 1]
        return np.bincount(codes, minlength=4) / len(codes)
    ode_occ = occ_from_gaps(late)
    g2, _ = ode_traj(inst, w, dt=0.0025, steps=160_000)
    ode_occ_fine = occ_from_gaps(g2[-len(g2) // 4:][:, states])
    sa = {str(a): stochastic_track(inst, w, list(states), a, HORIZON, seed=seed)["pattern_occupancy"]
          for a in (0.002, 0.0005)}
    # product-of-codimension-1 baseline: each state's own two-sided Filippov weight
    alphas = [filippov_alpha(inst, s, theta, w)[0] for s in states]
    a0, a1 = alphas
    product = [(1 - a0) * (1 - a1), (1 - a0) * a1, a0 * (1 - a1), a0 * a1] if np.isfinite(alphas).all() else None
    return {"seed": seed, "w": w, "states": states.tolist(), "feasible": True, "feasible_ranges": feas,
            "ode_occ": ode_occ.tolist(), "ode_occ_fine": ode_occ_fine.tolist(), "sa_occ": sa,
            "product": product,
            "dist_ode": distance_to_feasible(H, N, ode_occ), "dist_ode_fine": distance_to_feasible(H, N, ode_occ_fine),
            "dist_sa": {k: distance_to_feasible(H, N, np.array(v)) for k, v in sa.items()}}


def _moving(args):
    r, beta = args
    inst = instance(r["seed"])
    t = stochastic_track(inst, r["w"], [r["state"]], 0.002, HORIZON, seed=r["seed"], beta=beta, frac=0.5,
                         window=5000)
    occ = np.array(t["window_occupancy"])
    late = occ[len(occ) // 2:]
    return {"seed": r["seed"], "w": r["w"], "beta": beta, "alpha_frozen": r["alpha"],
            "churn": t["churn"][0], "late_mixed_windows": float(np.mean((late > 0.02) & (late < 0.98))),
            "late_occupancy_mean": float(late.mean()), "window_occupancy": occ.tolist(), "gap_trace": t["gap_trace"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+", choices=["stepsize", "codim2", "moving"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--scan", type=int, default=3000)
    args = ap.parse_args()
    with Pool(args.workers) as pool:
        if "stepsize" in args.which:
            rows = pool.map(_stepsize, codim1_cases(60))
            json.dump(rows, open(os.path.join(RES, "ext_stepsize.json"), "w"))
            print("stepsize done", flush=True)
        if "codim2" in args.which:
            rows = [r for r in pool.imap_unordered(_scan, [(s, w) for w in (1.0, 1.3) for s in range(args.scan)],
                                                   chunksize=8) if r is not None]
            json.dump(rows, open(os.path.join(RES, "ext_codim2.json"), "w"))
            print(f"codim2 done: {len(rows)} two-state chattering cases", flush=True)
        if "moving" in args.which:
            cases = codim1_cases(30, seed=1)
            rows = pool.map(_moving, [(r, b) for r in cases for b in (0.0, 1e-5, 1e-4)])
            json.dump(rows, open(os.path.join(RES, "ext_moving.json"), "w"))
            print("moving done", flush=True)


if __name__ == "__main__":
    main()
