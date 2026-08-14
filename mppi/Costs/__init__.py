"""Example costs for MPPI (pure PyTorch).

These are reference implementations for wiring ``MPPI(Dynamics, Costs, Sampling)``.
Production HOUND tracking uses ``hound_nav.trackingCostCUDA.SimpleCarCost`` (CUDA).

Expected Cost API used by ``mppi.MPPI``:
  - ``forward(states, controls) -> [K]`` trajectory costs
  - optional: ``set_BEV(...)``, ``set_path(...)``, ``set_goal(...)``
"""

from .SimpleCarCost import SimpleCarCost
from .TrackingCost import TrackingCost

__all__ = ["SimpleCarCost", "TrackingCost"]
