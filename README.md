# mppi

Standalone **MPPI** control library extracted from `BeamNGRL.control.UW_mppi`.

## Contents

- `mppi.MPPI` — Williams-style MPPI shell
- `mppi.Sampling.Delta_Sampling` — delta-control sampling (trailing `control_dim` state channels)
- `mppi.Dynamics.SimpleCarDynamicsTCUDA` — CUDA bicycle + BEV (`analytical_bicycle.{cu,cpp}` next to the module)
- `mppi.Costs` — **example** pure-torch costs (not used by HOUND deployment)
  - `SimpleCarCost` — goal-seeking (`set_goal` / `set_BEV`)
  - `TrackingCost` — path tracking (`set_path` / `set_BEV`)
- `mppi/Configs/*.yaml` — sample Dynamics / Sampling / Cost / Map / MPPI configs

HOUND deployment tracking cost is `hound_nav.trackingCostCUDA.SimpleCarCost` (CUDA).
Swap that in the same `MPPI(Dynamics, Costs, Sampling, …)` constructor.

## Cost API (for custom costs)

```text
forward(states, controls) -> [K]     # M×K×T×NX states, M×K×T×nu controls
set_BEV(height, normal, cost)        # optional
set_goal(xy) / set_path(T×4)         # optional
```

## Example

```bash
# after colcon build + source install/setup.bash
python3 /path/to/mppi/examples/wire_mppi_cost.py --cost tracking
python3 /path/to/mppi/examples/wire_mppi_cost.py --cost goal
```

## NX / plant size

Pass `Dynamics_config["NX"]` (or `state_dims` / `control_state_dims`). Default
and minimum is **17** (indices 0..14 used by the bicycle kernel; last 2 = actuators).

## Build

```bash
cd ~/colcon_ws && colcon build --packages-select mppi hound_nav --symlink-install
```
