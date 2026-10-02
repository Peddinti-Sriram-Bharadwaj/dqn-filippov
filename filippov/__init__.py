"""Sliding on action-tie manifolds in Double(-SOR) Q-learning, and its Filippov occupancy."""

from .dynamics import detect_sliding, drift, filippov_alpha, instance, ode, stochastic

__all__ = ["instance", "drift", "ode", "detect_sliding", "filippov_alpha", "stochastic"]
