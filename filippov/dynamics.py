"""Double-SOR Q-learning with linear features and a frozen target network: expected drift, its
deterministic (Euler) and stochastic (constant-step) dynamics, detection of sliding on action-tie
manifolds, and the Filippov weight predicting how sliding time splits between the two sides.

Q_theta(s, a) = phi(s, a)' theta. With the target parameters theta_bar fixed (the inner loop between
target synchronisations), the expected drift is

    h(theta) = sum_{s,a} d(s,a) phi(s,a) [y_w(s,a; theta) - Q_theta(s,a)],
    y_w = w (r + gamma sum_s' P(s'|s,a) Qbar(s', argmax_b Q_theta(s',b))) + (1 - w) Qbar(s, argmax_c Q_theta(s,c)),

with w = 1 giving Double Q-learning's target and w != 1 Double SOR's. The argmax makes h jump
across each action-tie hyperplane M_s = {Q_theta(s,1) = Q_theta(s,2)}.
"""

import numpy as np

S, A, K, GAMMA = 6, 2, 4, 0.9


def instance(seed):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(S) * 0.5, size=(S, A))           # [S, A, S]
    R = rng.normal(size=(S, A))
    phi = rng.normal(size=(S, A, K)) / np.sqrt(K)
    d = rng.dirichlet(np.ones(S * A)).reshape(S, A)              # behaviour weighting, uneven on purpose
    theta_bar = rng.normal(size=K)
    theta0 = rng.normal(size=K)
    return {"P": P, "R": R, "phi": phi, "d": d, "theta_bar": theta_bar, "theta0": theta0}


def greedy(phi, theta):
    q = phi @ theta
    return (q[:, 1] > q[:, 0]).astype(int)                       # action index per state


def drift(inst, theta, w, force=None):
    """Expected drift; force = (state, action) overrides the argmax at that state."""
    phi, P, R, d = inst["phi"], inst["P"], inst["R"], inst["d"]
    g = greedy(phi, theta)
    if force is not None:
        g = g.copy(); g[force[0]] = force[1]
    qbar = phi @ inst["theta_bar"]                                # [S, A]
    v = qbar[np.arange(S), g]                                     # Qbar(s, argmax_online)
    y = w * (R + GAMMA * P @ v) + (1 - w) * v[:, None]
    q = phi @ theta
    return np.einsum("sa,sak->k", d * (y - q), phi)


def ode(inst, w, dt=0.01, steps=40_000):
    theta = inst["theta0"].copy()
    gaps = np.zeros((steps, S))
    thetas = np.zeros((steps, K))
    for t in range(steps):
        theta = theta + dt * drift(inst, theta, w)
        q = inst["phi"] @ theta
        gaps[t] = q[:, 1] - q[:, 0]
        thetas[t] = theta
    return gaps, thetas


def detect_sliding(gaps, thetas, frac=0.25, min_flips=20):
    late = gaps[-int(len(gaps) * frac):]
    flips = (np.diff(np.sign(late), axis=0) != 0).sum(0)
    chattering = np.flatnonzero(flips >= min_flips)
    if len(chattering) != 1:
        return None
    th = thetas[-int(len(thetas) * frac):]
    if np.linalg.norm(th[-1] - th[len(th) // 2]) > 1e-2:     # sliding to rest, not still travelling
        return None
    return int(chattering[0]), th.mean(0)


def filippov_alpha(inst, s, theta, w):
    n = inst["phi"][s, 1] - inst["phi"][s, 0]                  # gradient of the gap at s
    theta_m = theta - (n @ theta) / (n @ n) * n                 # project onto M_s
    h_plus = drift(inst, theta_m, w, force=(s, 1))              # '+' side: action 1 greedy
    h_minus = drift(inst, theta_m, w, force=(s, 0))
    hp, hm = n @ h_plus / np.linalg.norm(n), n @ h_minus / np.linalg.norm(n)
    attracting = hp < 0 < hm                                    # both sides push into M_s
    return (abs(hm) / (abs(hp) + abs(hm)) if attracting else float("nan")), attracting, hp, hm


def stochastic(inst, w, s_star, step=2e-3, steps=300_000, seed=0, frac=0.5):
    rng = np.random.default_rng(seed)
    phi, P, R, d = inst["phi"], inst["P"], inst["R"], inst["d"]
    qbar = phi @ inst["theta_bar"]
    flat_d = d.reshape(-1)
    idx = rng.choice(S * A, size=steps, p=flat_d)
    u = rng.random(steps)
    cdf = P.cumsum(-1)
    theta = inst["theta0"].copy()
    late_start = int(steps * (1 - frac))
    on_plus = flips = 0
    prev = None
    for t in range(steps):
        s, a = divmod(int(idx[t]), A)
        s2 = min(int(np.searchsorted(cdf[s, a], u[t])), S - 1)
        q_s2 = phi[s2] @ theta
        q_s = phi[s] @ theta
        y = w * (R[s, a] + GAMMA * qbar[s2, int(q_s2[1] > q_s2[0])]) + (1 - w) * qbar[s, int(q_s[1] > q_s[0])]
        theta = theta + step * (y - q_s[a]) * phi[s, a]
        if t >= late_start:
            g = (phi[s_star] @ theta)
            cur = int(g[1] > g[0])
            on_plus += cur
            flips += prev is not None and cur != prev
            prev = cur
    n_late = steps - late_start
    return on_plus / n_late, flips / n_late


