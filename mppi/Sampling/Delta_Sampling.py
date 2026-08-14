import torch
import torch.nn as nn


class Delta_Sampling(torch.nn.Module):
    """MPPI delta-control sampling in physical units.

    Channel 0: steering (rad), symmetric ``±steer_lim``
    Channel 1: wheelspeed (m/s), ``[min_spd, max_spd]``
    Trailing ``control_dim`` plant-state channels hold the same units.
    """

    def __init__(
        self,
        sampling_config,
        MPPI_config,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):
        super(Delta_Sampling, self).__init__()
        self.dtype = dtype
        self.d = device

        self.nu = int(sampling_config["control_dim"])
        self.K = MPPI_config["ROLLOUTS"]
        self.T = MPPI_config["TIMESTEPS"]
        self.M = MPPI_config["BINS"]

        self.temperature = torch.tensor(
            sampling_config["temperature"], dtype=self.dtype, device=self.d
        )
        self.scaled_dt = torch.tensor(
            sampling_config["scaled_dt"], dtype=self.dtype, device=self.d
        )

        self.CTRL_NOISE = torch.zeros((self.nu, self.nu), device=self.d, dtype=self.dtype)
        self.CTRL_NOISE[0, 0] = float(sampling_config["noise_0"])
        self.CTRL_NOISE[1, 1] = float(sampling_config["noise_1"])

        self.CTRL_NOISE_inv = torch.inverse(self.CTRL_NOISE)
        self.CTRL_NOISE_MU = torch.zeros(self.nu, dtype=self.dtype, device=self.d)

        self.reset()
        self.noise = (
            torch.matmul(
                torch.randn((self.K, self.T, self.nu), device=self.d, dtype=self.dtype),
                self.CTRL_NOISE,
            )
            + self.CTRL_NOISE_MU
        )

        # Prefer physical-unit keys; fall back to legacy normalized throttle names.
        if "steer_lim" in sampling_config:
            self.steer_lim = float(sampling_config["steer_lim"])
        else:
            self.steer_lim = 1.0
        if "max_spd" in sampling_config:
            self.max_spd = float(sampling_config["max_spd"])
            self.min_spd = float(sampling_config.get("min_spd", 0.0))
        else:
            self.max_spd = float(sampling_config.get("max_thr", 1.0))
            self.min_spd = float(sampling_config.get("min_thr", 0.0))
        self.cost_total = 0

    def reset(self):
        torch.manual_seed(0)

    def _u0(self, state):
        """Actuator channels = trailing ``nu`` dims of the plant state."""
        return state[..., -self.nu :]

    def _clamp_controls(self, controls):
        controls[..., 0] = torch.clamp(controls[..., 0], -self.steer_lim, self.steer_lim)
        controls[..., 1] = torch.clamp(controls[..., 1], self.min_spd, self.max_spd)
        return controls

    def sample(self, state, U):
        self.noise = (
            torch.matmul(
                torch.randn((self.K, self.T, self.nu), device=self.d, dtype=self.dtype),
                self.CTRL_NOISE,
            )
            + self.CTRL_NOISE_MU
        )

        perturbed_actions = U + self.noise
        u0 = self._u0(state)
        controls = self._clamp_controls(
            u0
            + (self.scaled_dt)
            * torch.cumsum(perturbed_actions.unsqueeze(dim=0), dim=-2)
        )

        perturbed_actions[:, 1:, :] = torch.diff(controls - u0, dim=-2).squeeze(
            dim=0
        ) / (self.scaled_dt)

        self.noise = perturbed_actions - U

        action_cost = self.temperature * torch.matmul(self.noise, self.CTRL_NOISE_inv)
        perturbation_cost = torch.sum(U * action_cost, dim=(1, 2))

        return controls, perturbation_cost

    def update_control(self, cost_total, U, state):
        beta = torch.min(cost_total)
        self.cost_total = cost_total.clone()
        cost_total_non_zero = torch.exp((-1 / self.temperature) * (cost_total - beta))

        eta = torch.sum(cost_total_non_zero)
        omega = (1.0 / eta) * cost_total_non_zero

        U = U + (omega.view(-1, 1, 1) * self.noise).sum(dim=0)
        u0 = self._u0(state)
        controls = self._clamp_controls(u0 + self.scaled_dt * torch.cumsum(U, dim=-2))
        return controls, U
