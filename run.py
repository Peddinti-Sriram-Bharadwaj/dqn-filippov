"""Runs the experiment: random instances x w, writes results/results.jsonl and prints a summary.

    python run.py                                  # 300 instances x 4 values of w, ~15 min on 3 cores
    python run.py --instances 20 --workers 2       # quick check
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from filippov import detect_sliding, filippov_alpha, instance, ode, stochastic

HERE = os.path.dirname(os.path.abspath(__file__))


def run(args):
    seed, w = args
    inst = instance(seed)
    gaps, thetas = ode(inst, w)
    found = detect_sliding(gaps, thetas)
    out = {"seed": seed, "w": w, "sliding": found is not None}
    if found is None:
        return out
    s, theta = found
    alpha, attracting, hp, hm = filippov_alpha(inst, s, theta, w)
    late = gaps[-len(gaps) // 4:, s]
    out.update({"state": s, "alpha": alpha, "attracting": bool(attracting), "h_plus_n": float(hp),
                "h_minus_n": float(hm), "ode_occupancy": float((late > 0).mean())})
    if attracting:
        out["sa_occupancy"], out["sa_churn"] = stochastic(inst, w, s, seed=seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=300)
    ap.add_argument("--ws", nargs="+", type=float, default=[0.7, 1.0, 1.3, 1.6])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "results.jsonl"))
    args = ap.parse_args()
    tasks = [(s, w) for w in args.ws for s in range(args.instances)]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with Pool(args.workers) as pool, open(args.out, "w") as f:
        for r in pool.imap_unordered(run, tasks, chunksize=4):
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}; run `python analyze.py` for tables and figures")


if __name__ == "__main__":
    main()
