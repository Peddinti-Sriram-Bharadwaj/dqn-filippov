"""Codimension-2 selection at small step sizes: for every case in results/ext_codim2.json, the
stochastic sign-pattern occupancy at a = 0.002, 0.0005, 0.0002 and 0.0001 with four noise seeds each.
Writes results/ext_codim2_small.json.

    python run_codim2_small.py --workers 3
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from filippov.dynamics import instance
from filippov.extensions import stochastic_batch

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [0.002, 0.0005, 0.0002, 0.0001]
SEEDS = [11, 12, 13, 14]


def run(case):
    inst = instance(case["seed"])
    occ = {str(a): stochastic_batch(inst, case["w"], case["states"], a, 600.0, SEEDS).tolist() for a in STEPS}
    return {"seed": case["seed"], "w": case["w"], "states": case["states"], "occ": occ}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    cases = [c for c in json.load(open(os.path.join(HERE, "results", "ext_codim2.json"))) if c["feasible"]]
    path = os.path.join(HERE, "results", "ext_codim2_small.json")
    out = json.load(open(path)) if os.path.exists(path) else []  # resume
    done = {(r["seed"], r["w"]) for r in out}
    cases = [c for c in cases if (c["seed"], c["w"]) not in done]
    print(f"{len(done)} cases done, {len(cases)} to run", flush=True)
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run, cases), 1):
            out.append(r)
            json.dump(out, open(path, "w"))
            print(f"[{len(done) + i}/{len(done) + len(cases)}] seed {r['seed']} w {r['w']}", flush=True)


if __name__ == "__main__":
    main()
