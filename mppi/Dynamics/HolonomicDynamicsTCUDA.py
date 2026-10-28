#!/usr/bin/env python3
"""MPPI dynamics: holonomic_6D.cu integration. Controls are body vx, vy, omega."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def _load_kernel():
    dyn_dir = Path(__file__).resolve().parent
    sources = [str(dyn_dir / "holonomic_mppi.cpp"), str(dyn_dir / "holonomic_mppi.cu")]
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.7")
    try:
        from hound_nav.cuda_cache import load_extension

        return load_extension(
            "mppi_holonomic_6d", sources, extra_cuda_cflags=["-O3", "--use_fast_math"]
        )
    except ImportError:
        from torch.utils.cpp_extension import load

        return load(
            name="mppi_holonomic_6d",
            sources=sources,
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _load_kernel()
    return _KERNEL


class HolonomicDynamics:
    def __init__(
        self,
        Dynamics_config,
        Map_config,
        MPPI_config,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):
        if device.type != "cuda":
            raise RuntimeError("HolonomicDynamics requires CUDA")
        self.dtype = dtype
        self.d = device
        self.dt = np.float32(Dynamics_config["dt"])
        self.K = np.int32(MPPI_config["ROLLOUTS"])
        self.T = np.int32(MPPI_config["TIMESTEPS"])
        self.M = np.int32(MPPI_config["BINS"])
        self.NX = np.int32(int(Dynamics_config.get("NX", Dynamics_config.get("state_dims", 17))))
        self.NC = np.int32(int(Dynamics_config.get("NC", 3)))
        if int(self.NX) < 15 or int(self.NC) < 3:
            raise ValueError(f"HolonomicDynamics needs NX>=15 and NC>=3 (got {self.NX}, {self.NC})")
        self.BEVmap_size = np.float32(Map_config["map_size"])
        self.BEVmap_res = np.float32(Map_config["map_res"])
        self.BEVmap_size_px = np.int32(self.BEVmap_size / self.BEVmap_res)
        self.car_l2 = np.float32(Dynamics_config["car_length"] / 2)
        self.car_w2 = np.float32(Dynamics_config["car_width"] / 2)
        self.states = torch.zeros(
            (self.M, self.K, self.T, self.NX), dtype=self.dtype, device=self.d
        )
        self.BEVmap_height = torch.zeros(
            (self.BEVmap_size_px, self.BEVmap_size_px), dtype=self.dtype, device=self.d
        )
        _kernel()

    def set_BEV(self, BEVmap_height, BEVmap_normal):
        del BEVmap_normal
        self.BEVmap_height = BEVmap_height

    def forward(self, state, controls):
        state_ = state.squeeze(0)
        controls_ = controls.squeeze(0)
        _kernel().rollout(
            state_,
            controls_,
            self.BEVmap_height,
            float(self.dt),
            int(self.K),
            int(self.T),
            int(self.NX),
            int(self.NC),
            int(self.BEVmap_size_px),
            float(self.BEVmap_res),
            float(self.car_l2),
            float(self.car_w2),
        )
        self.states = torch.clone(state_).unsqueeze(0)
        return self.states
