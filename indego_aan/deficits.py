"""Deficit models for the simulated person (plan Phase 2): low volitional drive and muscle fatigue.

Both act on the person's muscle commands in indego_sim's HumanController without changing it: after each human tick
the muscle excitation is recomputed as

    u = min(drive(t) * a_voluntary + e_reflex, 1 - MF)

where `drive` scales the voluntary (supraspinal) part only - the spinal reflexes keep working - and MF is the
fatigued fraction of each muscle's motor units from the 3CC-r model. The person's own controller does not know
about either deficit (a "naive" person, plan Phase 3).

3CC-r (Xia & Frey Law 2008; recovery multiplier at rest: Looft et al. 2018, Rakshit et al. 2021): each muscle's
motor units are active (MA), resting (MR) or fatigued (MF), MA + MR + MF = 1,

    dMF/dt = F * MA - rR * MF,     rR = r * R at rest (command <= 0.01), R otherwise,

with MA the muscle's activation (MuJoCo's activation dynamics play the role of the 3CC-r transfer rates LD, LR, as in
MyoSuite's CumulativeFatigue, whose parameter table this module ports - Apache-2.0, MyoSuite Authors). A muscle
cannot be excited above its unfatigued fraction 1 - MF. `rate_scale` multiplies F and R (time compression: the
parameters come from sustained contractions, so their size for walking is calibrated in Phase 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from indego_sim.config import MUSCLE_GROUPS

# F, R in 1/s, r dimensionless: MyoSuite's MUSCLE_FATIGUE_PARAMS (Rakshit et al. 2021 J Biomech 127:110695, Table 2;
# "Default": Frey-Law et al. 2012 general F, R with r of Looft et al. 2018)
FATIGUE_PARAMS = {
    "Knee-Extensor": (0.00825, 0.00076, 14.85),
    "Knee": (0.00825, 0.00076, 14.85),
    "Ankle-Plantarflexor": (0.00695, 0.00096, 15.0),
    "Ankle-Dorsiflexor": (0.00828, 0.00204, 7.07),
    "Default": (0.00970, 0.00091, 15.0),
}
GROUP_FATIGUE = {"knee_extensors": "Knee-Extensor", "knee_flexors": "Knee", "ankle_plantarflexors": "Ankle-Plantarflexor",
                 "ankle_dorsiflexors": "Ankle-Dorsiflexor"}
REST = 0.01                  # command at or below this counts as rest (MyoSuite FATIGUE_REST_THRESHOLD)


@dataclass
class DriveProcess:
    """Volitional drive scale in [low, 1]: a level that drifts within the episode (Ornstein-Uhlenbeck around the
    mean) plus brief lapses. `schedule` [(t, value), ...] replaces the constant mean
    with a piecewise-linear profile (scripted volition drops); drift and lapses still apply on top."""
    mean: float = 1.0
    low: float = 0.3
    drift_sd: float = 0.0               # stationary SD of the drift
    drift_tau: float = 5.0              # s correlation time of the drift
    lapse_rate: float = 0.0             # lapses per s
    lapse_depth: float = 0.6            # drive drops by this fraction during a lapse
    lapse_dur: tuple = (0.2, 0.6)       # s
    schedule: Optional[list] = None
    seed: int = 0
    _x: float = field(default=0.0, init=False)
    _lapse_until: float = field(default=-1.0, init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def value(self, t: float, dt: float) -> float:
        mean = self.mean
        if self.schedule:
            ts, vs = zip(*self.schedule)
            mean = float(np.interp(t, ts, vs))
        if self.drift_sd > 0:
            a = dt / self.drift_tau
            self._x += -a * self._x + self.drift_sd * np.sqrt(2 * a) * self.rng.standard_normal()
        if self.lapse_rate > 0 and t > self._lapse_until and self.rng.random() < self.lapse_rate * dt:
            self._lapse_until = t + self.rng.uniform(*self.lapse_dur)
        v = mean + self._x
        if t < self._lapse_until:
            v *= 1.0 - self.lapse_depth
        return float(np.clip(v, self.low, 1.0))


class Fatigue3CCr:
    """3CC-r fatigued fraction MF per muscle (see module docstring)."""

    def __init__(self, muscle_names, rate_scale: float = 1.0, group_scale: Optional[Dict[str, float]] = None):
        self.names = list(muscle_names)
        n = len(self.names)
        self.F, self.R, self.r = np.zeros(n), np.zeros(n), np.zeros(n)
        group_scale = group_scale or {}
        self.group = ["other"] * n
        for i, nm in enumerate(self.names):
            base = nm[:-2] if nm[-2:] in ("_l", "_r") else nm
            grp = next((g for g, ms in MUSCLE_GROUPS.items() if base in ms), "other")
            self.group[i] = grp
            F, R, r = FATIGUE_PARAMS[GROUP_FATIGUE.get(grp, "Default")]
            s = rate_scale * group_scale.get(grp, 1.0)
            self.F[i], self.R[i], self.r[i] = F * s, R * s, r
        self.MF = np.zeros(n)

    def reset(self, MF: Optional[np.ndarray] = None):
        self.MF = np.zeros(len(self.names)) if MF is None else np.asarray(MF, float).copy()

    def limit(self, u: np.ndarray, act: np.ndarray, dt: float) -> np.ndarray:
        """Advance MF over dt with the current activations `act` (MA) and commands `u` (TL); return u capped at
        the unfatigued fraction."""
        MA = np.clip(act, 0.0, 1.0 - self.MF)
        rR = np.where(u <= REST, self.r * self.R, self.R)
        self.MF = np.clip(self.MF + (self.F * MA - rR * self.MF) * dt, 0.0, 1.0)
        return np.minimum(u, 1.0 - self.MF)

    def group_mean(self) -> Dict[str, float]:
        if not hasattr(self, "_gidx"):
            self._gidx = {g: np.array([i for i, gg in enumerate(self.group) if gg == g]) for g in set(self.group)}
        return {g: float(self.MF[i].mean()) for g, i in self._gidx.items()}


class Deficits:
    """Attaches a drive process and/or fatigue to a HumanExoSim's person (wraps sim.human.step).

    mode "capacity" (default): the deficits limit what the person can recruit - their static optimisation shares the
    load among the muscles within drive x (1 - MF) of each muscle's capacity, so they compensate where they can
    (reduced maximal voluntary activation for low drive, as in central fatigue; fatigued motor units unavailable).
    mode "command": the person plans as if fresh and the commands are scaled by the drive and capped by fatigue
    afterwards - no compensation at all."""

    def __init__(self, sim, drive: Optional[DriveProcess] = None, fatigue: Optional[Fatigue3CCr] = None,
                 mode: str = "capacity"):
        if mode not in ("capacity", "command"):
            raise ValueError("mode must be 'capacity' or 'command'")
        self.sim, self.drive, self.fatigue, self.mode = sim, drive, fatigue, mode
        hc = sim.human
        self.hc = hc
        self.act_adr = sim.m.actuator_actadr[hc.mm.ids]
        self.drive_now = 1.0
        self._orig = hc.step
        hc.step = self._step
        self._orig_so = hc.mm.static_optimisation
        if mode == "capacity":
            hc.mm.static_optimisation = self._so

    def _available(self):
        a = self.drive_now if self.drive is not None else 1.0
        return a * (1.0 - self.fatigue.MF) if self.fatigue is not None else a

    def _so(self, d, tau_des, cap, **k):
        return self._orig_so(d, tau_des, cap * self._available(), **k)

    def _step(self, d, *a, **k):
        hc = self.hc
        self.drive_now = self.drive.value(d.time - self.sim.t0, hc.dt) if self.drive else 1.0
        it = self._orig(d, *a, **k)
        scale = self.drive_now if self.mode == "command" else 1.0
        u = np.clip(scale * hc.a_vol + hc.e_reflex, 0.0, 1.0)
        if self.fatigue is not None:
            u = self.fatigue.limit(u, d.act[self.act_adr], hc.dt)
        d.ctrl[hc.mm.ids] = u
        return it

    def detach(self):
        self.hc.step = self._orig
        self.hc.mm.static_optimisation = self._orig_so


class CommandDeficits:
    """Deficits for a person whose controller outputs the muscle commands itself (the learned KINESIS human,
    indego_aan.kinesis_human): its commands are scaled by the drive and capped by fatigue; it does not re-plan, but
    its policy reacts to the resulting tracking errors through feedback (the "command" mode of Deficits)."""

    def __init__(self, sim, drive: Optional[DriveProcess] = None, fatigue: Optional[Fatigue3CCr] = None):
        self.sim, self.drive, self.fatigue = sim, drive, fatigue
        self.act_adr = sim.m.actuator_actadr[sim.human.mm.ids]
        self.drive_now = 1.0

    def apply(self, d, u: np.ndarray, dt: float) -> np.ndarray:
        self.drive_now = self.drive.value(d.time - self.sim.t0, dt) if self.drive else 1.0
        u = np.clip(self.drive_now * u, 0.0, 1.0)
        if self.fatigue is not None:
            u = self.fatigue.limit(u, d.act[self.act_adr], dt)
        return u
