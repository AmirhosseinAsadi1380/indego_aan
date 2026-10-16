"""Non-learned exo controllers for comparison with the learned policy.

PersonalAAN: assist-as-needed toward the person's OWN fresh gait (plan: "keep a person walking close to their own
normal gait ... giving only the assistance that is missing"). Reference: the person's fresh mean hip and knee curves
over the gait cycle as the exo encoders see them (lab session B / `scripts/fresh_reference.py`). The gait phase is
found by matching the current hip and knee angles and velocities of both legs to that cycle, searched around the
phase predicted from the last estimate and the fresh stride time. The assistance stiffness of each joint starts at
zero and grows only while the tracking error is beyond `band` x the person's fresh step-to-step SD at that phase; it
decays otherwise (Emken / Banala-style forgetting). Off while the person stands (no hip moving). Uses only Indego
sensors (encoders, firmware velocity).

Privileged variant (Phase 4 teacher): with `deficit_gain` > 0 and `capacity` set from the true deficit each step
(drive x (1 - hip/knee fatigue), 1 = fresh), the stiffness never falls below deficit_gain x (1 - capacity): the
assistance is scheduled by the deficit as well as earned by tracking errors. Off by default (deployable version).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HIP_L, KNEE_L, HIP_R, KNEE_R = 0, 1, 2, 3


@dataclass
class PersonalAANParams:
    k_max: tuple = (60.0, 40.0, 60.0, 40.0)     # N*m/rad maximum assistance stiffness (hip_l, knee_l, hip_r, knee_r)
    d_ratio: float = 0.08                       # s: damping = d_ratio x stiffness
    band: float = 2.0                           # error dead band in fresh SDs: no assistance within it
    adapt_up: float = 6.0                       # 1/s per rad of error beyond the band
    forget: float = 0.15                        # 1/s gain decay
    tau_max: float = 40.0                       # N*m
    walk_hip_speed: float = 0.5                 # rad/s: stepping detected while a hip moved this fast in the last stride
    deficit_gain: float = 0.0                   # privileged: minimum stiffness fraction per unit of lost capacity


class PersonalAAN:
    def __init__(self, ref: dict, p: PersonalAANParams = None, dt: float = 0.005):
        """ref: dict(mean=(101, 4), sd=(101, 4) exo-encoder curves from right heel strike to right heel strike,
        step_time=s)."""
        self.p, self.dt = p or PersonalAANParams(), dt
        self.mean, self.sd = np.asarray(ref["mean"], float), np.asarray(ref["sd"], float)
        self.stride_time = 2.0 * float(ref["step_time"])
        self.ph_grid = np.linspace(0.0, 1.0, self.mean.shape[0])
        self.dmean = np.gradient(self.mean, self.ph_grid, axis=0) / self.stride_time
        # state for phase matching: angles and velocities (velocities scaled to radians of the cycle)
        self.ref_x = np.concatenate([self.mean, self.dmean * self.stride_time / (2 * np.pi)], axis=1)
        self.scale = np.maximum(np.ptp(self.ref_x, axis=0), 0.05)
        self.reset()

    def reset(self):
        self.capacity = 1.0           # privileged input (teacher): set by the caller each step
        self.k = np.zeros(4)
        self.phase = None
        self.hip_speed = 0.0          # peak hip speed over about the last stride (decaying envelope)

    def _phase(self, obs):
        """Phase in the reference cycle (0 = right heel strike) whose angles and velocities best match the current
        ones, within a window around the prediction (continuity; a slower or faster stride is followed)."""
        x = np.concatenate([obs["q"], np.asarray(obs["qd"]) * self.stride_time / (2 * np.pi)])
        cost = np.sum(((self.ref_x - x) / self.scale) ** 2, axis=1)
        if self.phase is None:
            i = int(np.argmin(cost))
        else:
            pred = (self.phase + self.dt / self.stride_time) % 1.0
            d = (self.ph_grid - pred + 0.5) % 1.0 - 0.5                    # signed phase distance to the prediction
            cost = cost + np.where((d > -0.08) & (d < 0.15), 0.0, np.inf)
            i = int(np.argmin(cost))
        self.phase = self.ph_grid[i]
        return self.phase

    def _ref(self, curve, ph, c):
        return float(np.interp(ph, self.ph_grid, curve[:, c], period=1.0))

    def __call__(self, obs):
        p = self.p
        ph = self._phase(obs)
        q, qd = np.asarray(obs["q"], float), np.asarray(obs["qd"], float)
        q_ref = np.array([self._ref(self.mean, ph, c) for c in range(4)])
        qd_ref = np.array([self._ref(self.dmean, ph, c) for c in range(4)])
        sd = np.array([self._ref(self.sd, ph, c) for c in range(4)])
        self.hip_speed = max(self.hip_speed * np.exp(-self.dt / self.stride_time), abs(qd[HIP_L]), abs(qd[HIP_R]))
        if self.hip_speed < p.walk_hip_speed:      # standing: no gait reference, no assistance, no adaptation
            self.k *= np.exp(-p.forget * self.dt)
            return np.zeros(4)
        e = q_ref - q
        beyond = np.sign(e) * np.maximum(np.abs(e) - p.band * sd, 0.0)
        self.k = np.clip(self.k + self.dt * (p.adapt_up * np.abs(beyond) - p.forget * self.k), 0.0, 1.0)
        if p.deficit_gain > 0:
            self.k = np.maximum(self.k, min(p.deficit_gain * (1.0 - self.capacity), 1.0))
        K = self.k * np.array(p.k_max)
        tau = K * beyond + p.d_ratio * K * (qd_ref - qd)
        return np.clip(tau, -p.tau_max, p.tau_max)
