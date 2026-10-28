"""MPPI sampling in delta body vx, vy, omega. Same cumsum as Delta_Sampling."""

import torch
import torch.nn as nn


class HolonomicDeltaSampling(torch.nn.Module):
    def __init__(
        self,
        sampling_config,
        MPPI_config,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):
        super().__init__()
        self.dtype = dtype
        self.d = device
        self.nu = int(sampling_config["control_dim"])
        if self.nu != 3:
            raise ValueError(f"HolonomicDeltaSampling control_dim must be 3 (got {self.nu})")
        self.K = MPPI_config["ROLLOUTS"]
        self.T = MPPI_config["TIMESTEPS"]
        self.temperature = torch.tensor(
            sampling_config["temperature"], dtype=self.dtype, device=self.d
        )
        self.scaled_dt = torch.tensor(
            sampling_config["scaled_dt"], dtype=self.dtype, device=self.d
        )
        self.CTRL_NOISE = torch.zeros((self.nu, self.nu), device=self.d, dtype=self.dtype)
        for i in range(self.nu):
            self.CTRL_NOISE[i, i] = float(sampling_config[f"noise_{i}"])
        self.CTRL_NOISE_inv = torch.inverse(self.CTRL_NOISE)
        self.CTRL_NOISE_MU = torch.zeros(self.nu, dtype=self.dtype, device=self.d)
        self.min_vel = float(sampling_config["min_vel"])
        self.max_vel = float(sampling_config["max_vel"])
        self.vy_max = float(sampling_config["vy_max"])
        self.max_omega = float(sampling_config["max_omega"])
        self.noise = torch.zeros((self.K, self.T, self.nu), device=self.d, dtype=self.dtype)
        self.cost_total = 0

    def reset(self):
        torch.manual_seed(0)

    def _u0(self, state):
        # Bicycle/FCU layout: vx, vy, yaw rate. Not the trailing command slots.
        return torch.stack((state[..., 6], state[..., 7], state[..., 14]), dim=-1)

    def _clamp(self, controls):
        controls[..., 0] = torch.clamp(controls[..., 0], self.min_vel, self.max_vel)
        controls[..., 1] = torch.clamp(controls[..., 1], -self.vy_max, self.vy_max)
        controls[..., 2] = torch.clamp(controls[..., 2], -self.max_omega, self.max_omega)
        return controls

    def sample(self, state, U):
        self.noise = (
            torch.matmul(
                torch.randn((self.K, self.T, self.nu), device=self.d, dtype=self.dtype),
                self.CTRL_NOISE,
            )
            + self.CTRL_NOISE_MU
        )
        perturbed = U + self.noise
        u0 = self._u0(state)
        controls = self._clamp(
            u0 + self.scaled_dt * torch.cumsum(perturbed.unsqueeze(dim=0), dim=-2)
        )
        perturbed[:, 1:, :] = torch.diff(controls - u0, dim=-2).squeeze(dim=0) / self.scaled_dt
        self.noise = perturbed - U
        action_cost = self.temperature * torch.matmul(self.noise, self.CTRL_NOISE_inv)
        perturbation_cost = torch.sum(U * action_cost, dim=(1, 2))
        return controls, perturbation_cost

    def update_control(self, cost_total, U, state):
        beta = torch.min(cost_total)
        self.cost_total = cost_total.clone()
        weight = torch.exp((-1 / self.temperature) * (cost_total - beta))
        omega = weight / torch.sum(weight)
        U = U + (omega.view(-1, 1, 1) * self.noise).sum(dim=0)
        controls = self._clamp(self._u0(state) + self.scaled_dt * torch.cumsum(U, dim=-2))
        return controls, U


if __name__ == "__main__":
    cfg = {
        "control_dim": 3,
        "noise_0": 0.2,
        "noise_1": 0.1,
        "noise_2": 0.05,
        "scaled_dt": 0.1,
        "temperature": 0.1,
        "min_vel": -0.2,
        "max_vel": 1.5,
        "vy_max": 0.5,
        "max_omega": 1.0,
    }
    mppi = {"ROLLOUTS": 2, "TIMESTEPS": 4, "BINS": 1}
    sampler = HolonomicDeltaSampling(cfg, mppi, device=torch.device("cpu"))
    state = torch.zeros(1, 2, 4, 17)
    state[..., 6] = 10.0
    state[..., 7] = 5.0
    state[..., 14] = 5.0
    controls, _ = sampler.sample(state, torch.zeros(4, 3))
    assert controls.shape[-1] == 3, controls.shape
    assert torch.all(controls[..., 0] <= 1.5 + 1e-5)
    assert torch.all(controls[..., 0] >= -0.2 - 1e-5)
    assert torch.all(controls[..., 1].abs() <= 0.5 + 1e-5)
    assert torch.all(controls[..., 2].abs() <= 1.0 + 1e-5)
    print("ok")
