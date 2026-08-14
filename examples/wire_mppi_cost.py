#!/usr/bin/env python3
"""Wire MPPI with an example cost (no robot / no BeamNG).

Shows the Cost API expected by ``mppi.MPPI``:

  Costs.forward(states, controls) -> [K]
  optional: set_BEV / set_goal (SimpleCarCost) or set_path (TrackingCost)

Usage (after ``source install/setup.bash``)::

  python3 examples/wire_mppi_cost.py
  python3 examples/wire_mppi_cost.py --cost tracking
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from mppi import MPPI
from mppi.Costs import SimpleCarCost, TrackingCost
from mppi.Dynamics import SimpleCarDynamics
from mppi.Sampling import Delta_Sampling


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cost",
        choices=("goal", "tracking"),
        default="tracking",
        help="goal=SimpleCarCost, tracking=TrackingCost",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=64,
        help="Override ROLLOUTS for a quick smoke run",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for SimpleCarDynamicsTCUDA")

    cfg_dir = Path(__file__).resolve().parents[1] / "mppi" / "Configs"
    MPPI_config = _load_yaml(cfg_dir / "MPPI_config.yaml")
    Dynamics_config = _load_yaml(cfg_dir / "Dynamics_config.yaml")
    Sampling_config = _load_yaml(cfg_dir / "Sampling_config.yaml")
    Map_config = _load_yaml(cfg_dir / "Map_config.yaml")
    MPPI_config["ROLLOUTS"] = args.rollouts
    Dynamics_config.setdefault("NX", 17)

    device = torch.device("cuda")
    dtype = torch.float32

    dynamics = SimpleCarDynamics(
        Dynamics_config, Map_config, MPPI_config, dtype=dtype, device=device
    )
    sampling = Delta_Sampling(Sampling_config, MPPI_config, dtype=dtype, device=device)

    px = int(Map_config["map_size"] / Map_config["map_res"])
    height = torch.zeros((px, px), device=device, dtype=dtype)
    normal = torch.zeros((px, px, 3), device=device, dtype=dtype)
    normal[..., 2] = 1.0
    costmap = torch.full((px, px), 255.0, device=device, dtype=dtype)

    if args.cost == "goal":
        cost_cfg = _load_yaml(cfg_dir / "Cost_config.yaml")
        costs = SimpleCarCost(cost_cfg, Map_config, dtype=dtype, device=device)
        # SimpleCarCost expects a path/semantic BEV (HxWx3), not the tracking costmap.
        bev_path = torch.zeros((px, px, 3), device=device, dtype=dtype)
        costs.set_BEV(height, normal, bev_path)
        costs.set_goal(torch.tensor([5.0, 0.0], device=device, dtype=dtype))
    else:
        cost_cfg = _load_yaml(cfg_dir / "Tracking_Cost_config.yaml")
        costs = TrackingCost(cost_cfg, Map_config, dtype=dtype, device=device)
        costs.set_BEV(height, normal, costmap)
        T = MPPI_config["TIMESTEPS"]
        path = torch.zeros((T, 4), device=device, dtype=dtype)
        path[:, 0] = torch.linspace(0.0, 5.0, T, device=device)
        path[:, 3] = 5.0
        costs.set_path(path)

    controller = MPPI(dynamics, costs, sampling, MPPI_config, device=device, dtype=dtype)

    dynamics.set_BEV(height, normal)
    state = torch.zeros(17, device=device, dtype=dtype)
    state[11] = 9.8  # az ≈ g
    action = controller.forward(state)
    print(f"cost={args.cost}  action={action.detach().cpu().numpy().reshape(-1)}")


if __name__ == "__main__":
    main()
