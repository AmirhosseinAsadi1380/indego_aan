"""Instrumented rollouts: what the success criteria need, recorded at the exo control rate (ground truth, sim only)."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

JOINTS = ("hip_flexion", "knee_angle", "ankle_angle")
SIDES = ("l", "r")


class Recorder:
    """Per control step: human joint angles/velocities and muscle joint torques (hip, knee, ankle), exo joint angles
    (as the encoders see them, hip_l, knee_l, hip_r, knee_r), velocities and motor torques, COM, vertical foot loads,
    drive and mean fatigue."""

    def __init__(self, sim, deficits=None):
        self.sim, self.deficits = sim, deficits
        m, hc = sim.m, sim.human
        self.chan = [f"{j}_{s}" for s in SIDES for j in JOINTS]
        idx = [hc.dof_names.index(c) for c in self.chan]
        self.q_adr = hc.qadr[idx]
        self.d_adr = hc.dadr[idx]
        self.mm_rows = [hc.mm.dof_names.index(c) for c in self.chan]
        self.exo_dof = np.array([m.jnt_dofadr[m.joint(j).id] for j in sim.info.exo_joints])   # hip_l, knee_l, ...
        self.rows = {k: [] for k in ("t", "q", "qd", "tau_h", "q_exo", "qd_exo", "tau_exo", "com_x", "com_vx",
                                     "load", "drive", "fatigue")}
        self._com_prev = None

    def __call__(self):
        sim, d, hc = self.sim, self.sim.d, self.sim.human
        t = d.time - sim.t0
        com = float(d.subtree_com[sim.root][0])
        vx = 0.0 if self._com_prev is None else (com - self._com_prev) / sim.ctrl_dt
        self._com_prev = com
        tau_h = hc.mm.moment_arms(d)[self.mm_rows] @ d.actuator_force[hc.mm.ids]
        r = self.rows
        r["t"].append(t)
        r["q"].append(d.qpos[self.q_adr].copy())
        r["qd"].append(d.qvel[self.d_adr].copy())
        r["tau_h"].append(tau_h)
        r["q_exo"].append(sim.exo_angles())
        r["qd_exo"].append(d.qvel[self.exo_dof].copy())
        r["tau_exo"].append(d.qfrc_actuator[self.exo_dof].copy())
        r["com_x"].append(com)
        r["com_vx"].append(vx)
        r["load"].append([max(float(sim._grf[s][2]), 0.0) for s in SIDES])
        defi = self.deficits
        r["drive"].append(defi.drive_now if defi else 1.0)
        r["fatigue"].append(float(np.mean(defi.fatigue.MF)) if defi and defi.fatigue is not None else 0.0)

    def arrays(self) -> Dict[str, np.ndarray]:
        out = {k: np.asarray(v, float) for k, v in self.rows.items()}
        out["chan"] = self.chan
        out["dt"] = self.sim.ctrl_dt
        out["body_weight"] = self.sim.BW
        out["fell"] = self.sim.fell
        return out


def run_episode(sim, controller: Callable, T: float, deficits=None, obs=None) -> Dict[str, np.ndarray]:
    """Step `controller(obs) -> 4 exo torques` for T s (or until a fall); returns the recorded arrays."""
    rec = Recorder(sim, deficits)
    obs = sim.observe() if obs is None else obs
    for _ in range(int(round(T / sim.ctrl_dt))):
        obs = sim.step(controller(obs), record=False)
        rec()
        if sim.fell:
            break
    return rec.arrays()
