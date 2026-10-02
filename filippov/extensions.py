"""Extensions toward Borkar's (2026) new results: the small-step limit, codimension-2 sliding (where
Filippov's tangency conditions do not determine how time splits between sign patterns), and a
slowly moving target network (two timescales).
"""

import itertools

import numpy as np
from scipy.optimize import linprog

from .dynamics import A, GAMMA, K, S, drift, instance


def detect_chattering(gaps, thetas, frac=0.25, min_flips=20, rest_tol=1e-2):
    """States whose greedy action keeps flipping in the late window while theta is at rest."""
    late = gaps[-int(len(gaps) * frac):]
    flips = (np.diff(np.sign(late), axis=0) != 0).sum(0)
    th = thetas[-int(len(thetas) * frac):]
    at_rest = np.linalg.norm(th[-1] - th[len(th) // 2]) <= rest_tol
    return np.flatnonzero(flips >= min_flips), th.mean(0), at_rest


def ode_traj(inst, w, dt=0.01, steps=40_000):
    theta = inst["theta0"].copy()
    gaps, thetas = np.zeros((steps, S)), np.zeros((steps, K))
    for t in range(steps):
        theta = theta + dt * drift(inst, theta, w)
        q = inst["phi"] @ theta
        gaps[t], thetas[t] = q[:, 1] - q[:, 0], theta
    return gaps, thetas


def stochastic_track(inst, w, states, step, horizon, seed=0, frac=0.5, beta=0.0, window=None):
    """Constant-step stochastic Double-SOR Q-learning for horizon/step updates.

    Records, over the last `frac` of updates, the joint sign pattern of the gaps at `states`
    (occupancy of each pattern), and the churn at each state (greedy flips per update). With
    beta > 0 the target moves by Polyak averaging theta_bar += beta (theta - theta_bar) each update.
    With `window`, also returns per-window occupancies of the first state and the gap trace.
    """
    rng = np.random.default_rng(seed)
    phi, P, R, d = inst["phi"], inst["P"], inst["R"], inst["d"]
    steps = int(horizon / step)
    idx = rng.choice(S * A, size=steps, p=d.reshape(-1))
    u = rng.random(steps)
    cdf = P.cumsum(-1)
    theta, theta_bar = inst["theta0"].copy(), inst["theta_bar"].copy()
    qbar = phi @ theta_bar
    start = int(steps * (1 - frac))
    states = list(states)
    counts = np.zeros(2 ** len(states))
    flips = np.zeros(len(states))
    prev = None
    win_occ, win_cur, win_n, trace = [], 0, 0, []
    for t in range(steps):
        s, a = divmod(int(idx[t]), A)
        s2 = min(int(np.searchsorted(cdf[s, a], u[t])), S - 1)
        q_s2, q_s = phi[s2] @ theta, phi[s] @ theta
        y = w * (R[s, a] + GAMMA * qbar[s2, int(q_s2[1] > q_s2[0])]) + (1 - w) * qbar[s, int(q_s[1] > q_s[0])]
        theta = theta + step * (y - q_s[a]) * phi[s, a]
        if beta > 0:
            theta_bar = theta_bar + beta * (theta - theta_bar)
            if t % 50 == 0:
                qbar = phi @ theta_bar
        if t >= start:
            g = phi[states] @ theta                       # [len(states), A]
            cur = (g[:, 1] > g[:, 0]).astype(int)
            counts[int("".join(map(str, cur)), 2)] += 1
            if prev is not None:
                flips += cur != prev
            prev = cur
        if window:
            gs = phi[states[0]] @ theta
            win_cur += int(gs[1] > gs[0]); win_n += 1
            if win_n == window:
                win_occ.append(win_cur / window); win_cur = win_n = 0
                trace.append(float(gs[1] - gs[0]))
    n = steps - start
    out = {"pattern_occupancy": (counts / n).tolist(), "churn": (flips / n).tolist(), "steps": steps}
    if window:
        out["window_occupancy"], out["gap_trace"] = win_occ, trace
    return out


def pattern_drifts(inst, theta, w, states):
    """One-sided drifts for every sign pattern at `states`, at theta projected onto the intersection
    of their tie hyperplanes. Pattern bits are '1' when action 1 is greedy, ordered as `states`."""
    N = np.stack([inst["phi"][s, 1] - inst["phi"][s, 0] for s in states])     # normals [m, K]
    theta_m = theta - N.T @ np.linalg.solve(N @ N.T, N @ theta)
    H = []
    for bits in itertools.product((0, 1), repeat=len(states)):
        g_force = list(zip(states, bits))
        H.append(_drift_forced(inst, theta_m, w, g_force))
    return np.array(H), N


def _drift_forced(inst, theta, w, forced):
    phi, P, R, d = inst["phi"], inst["P"], inst["R"], inst["d"]
    q = phi @ theta
    g = (q[:, 1] > q[:, 0]).astype(int)
    for s, b in forced:
        g[s] = b
    qbar = phi @ inst["theta_bar"]
    v = qbar[np.arange(S), g]
    y = w * (R + GAMMA * P @ v) + (1 - w) * v[:, None]
    return np.einsum("sa,sak->k", d * (y - q), phi)


def filippov_feasible_set(H, N):
    """Convex weights lambda over sign patterns whose combined drift is tangent to the intersection:
    N (sum_q lambda_q h_q) = 0, sum lambda = 1, lambda >= 0. Returns, per pattern, the [min, max]
    of lambda_q over this set (None if empty). For codimension m there are 2^m unknowns and m + 1
    equations, so the set is generally not a single point."""
    nq = len(H)
    A_eq = np.vstack([(N @ H.T), np.ones((1, nq))])
    b_eq = np.r_[np.zeros(len(N)), 1.0]
    ranges = []
    for q in range(nq):
        lo = linprog(np.eye(nq)[q], A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)] * nq, method="highs")
        hi = linprog(-np.eye(nq)[q], A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)] * nq, method="highs")
        if not (lo.success and hi.success):
            return None
        ranges.append((lo.fun, -hi.fun))
    return ranges


def distance_to_feasible(H, N, occ):
    """Smallest L1 distance from an observed occupancy vector to the feasible Filippov set."""
    nq = len(H)
    # variables: lambda (nq), slack t (nq) with |lambda - occ| <= t; minimise sum t
    c = np.r_[np.zeros(nq), np.ones(nq)]
    A_eq = np.hstack([np.vstack([(N @ H.T), np.ones((1, nq))]), np.zeros((len(N) + 1, nq))])
    b_eq = np.r_[np.zeros(len(N)), 1.0]
    I = np.eye(nq)
    A_ub = np.vstack([np.hstack([I, -I]), np.hstack([-I, -I])])
    b_ub = np.r_[occ, -np.asarray(occ)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, 1)] * nq + [(0, None)] * nq, method="highs")
    return res.fun if res.success else float("nan")


__all__ = ["instance", "detect_chattering", "ode_traj", "stochastic_track", "pattern_drifts",
           "filippov_feasible_set", "distance_to_feasible"]


def stochastic_batch(inst, w, states, step, horizon, seeds, frac=0.5):
    """stochastic_track's sign-pattern occupancy for several noise seeds at once (vectorised over
    seeds). Returns an array [len(seeds), 2**len(states)]."""
    phi, P, R, d = inst["phi"], inst["P"], inst["R"], inst["d"]
    qbar = phi @ inst["theta_bar"]
    n, steps = len(seeds), int(horizon / step)
    rngs = [np.random.default_rng(s) for s in seeds]
    cdf = P.cumsum(-1)
    flat_d = d.reshape(-1)
    theta = np.tile(inst["theta0"], (n, 1))
    states = list(states)
    m = len(states)
    counts = np.zeros((n, 2 ** m))
    start = int(steps * (1 - frac))
    weights = 2 ** np.arange(m - 1, -1, -1)
    rows = np.arange(n)
    chunk = 100_000
    for c0 in range(0, steps, chunk):
        c1 = min(steps, c0 + chunk)
        idx = np.stack([r.choice(S * A, size=c1 - c0, p=flat_d) for r in rngs], 1)    # [chunk, n]
        u = np.stack([r.random(c1 - c0) for r in rngs], 1)
        for j in range(c1 - c0):
            s, a = np.divmod(idx[j], A)
            s2 = np.minimum((u[j][:, None] > cdf[s, a]).sum(1), S - 1)
            q_s2 = np.einsum("nak,nk->na", phi[s2], theta)
            q_s = np.einsum("nak,nk->na", phi[s], theta)
            g2 = (q_s2[:, 1] > q_s2[:, 0]).astype(int)
            g1 = (q_s[:, 1] > q_s[:, 0]).astype(int)
            y = w * (R[s, a] + GAMMA * qbar[s2, g2]) + (1 - w) * qbar[s, g1]
            theta += step * (y - q_s[rows, a])[:, None] * phi[s, a]
            if c0 + j >= start:
                g = np.einsum("nmak,nk->nma", phi[states][None].repeat(n, 0), theta)
                bits = (g[:, :, 1] > g[:, :, 0]).astype(int)
                counts[rows, bits @ weights] += 1
    return counts / (steps - start)
