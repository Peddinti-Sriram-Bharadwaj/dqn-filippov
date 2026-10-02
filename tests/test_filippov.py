import numpy as np

from filippov import detect_sliding, drift, filippov_alpha, instance, ode


def test_filippov_weight_one_dimensional():
    # x' = -1 above 0 and +3 below: sliding at 0 with time fraction 3 / (1 + 3) on the '+' side.
    x, dt, above = 0.5, 1e-3, []
    for _ in range(200_000):
        x += dt * (-1.0 if x > 0 else 3.0)
        above.append(x > 0)
    assert abs(np.mean(above[100_000:]) - 0.75) < 0.01


def test_forced_drift_matches_free_drift_off_the_manifold():
    inst = instance(0)
    theta = inst["theta0"]
    q = inst["phi"] @ theta
    s = 0
    natural = int(q[s, 1] > q[s, 0])
    assert np.allclose(drift(inst, theta, 1.0), drift(inst, theta, 1.0, force=(s, natural)))


def test_detected_sliding_satisfies_the_attraction_condition():
    inst = instance(4)
    gaps, thetas = ode(inst, 1.0)
    s, theta = detect_sliding(gaps, thetas)
    alpha, attracting, hp, hm = filippov_alpha(inst, s, theta, 1.0)
    assert attracting and hp < 0 < hm
    assert abs(np.mean(gaps[-10_000:, s] > 0) - alpha) < 0.01
