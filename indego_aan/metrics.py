"""The project's three success criteria, computed from recorded rollouts (indego_aan.rollout).

* Gait stays normal: joint-angle curves over the gait cycle (hip, knee, ankle, both sides, time-normalised between
  right heel strikes) compared with the person's fresh baseline - the deviation of the mean curve divided by the
  baseline's cycle-to-cycle variability (<= 1: within fresh variability) - plus speed and step-time variability.
  `gait_deviation` (the pass/fail number) is the worst of the joints the Indego drives (hip and knee, both sides);
  the ankles, which it cannot drive, are reported as `gait_deviation_ankle`.
* Assistance is minimal: mean squared exo torque and exo positive work per second (hip + knee).
* The human keeps working: the human's share of hip + knee positive work, human / (human + exo).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

N_PHASE = 101


def heel_strikes(rec, side: int, frac: float = 0.05, min_gap: float = 0.3, min_stance: float = 0.15) -> np.ndarray:
    """Indices where the vertical load on foot `side` (0 = left, 1 = right) rises above `frac` of body weight and stays
    there for most of the next `min_stance` s (a foot brushing the ground in swing is not a heel strike)."""
    on = rec["load"][:, side] > frac * rec["body_weight"]
    idx = np.flatnonzero(on[1:] & ~on[:-1]) + 1
    w = max(1, int(round(min_stance / rec["dt"])))
    keep, last = [], -np.inf
    for i in idx:
        if (i - last) * rec["dt"] >= min_gap and on[i:i + w].mean() >= 0.8:
            keep.append(i)
            last = i
    return _drop_early(np.asarray(keep, int))


def _drop_early(hs: np.ndarray, frac: float = 0.6) -> np.ndarray:
    """Drop strikes that come sooner than frac x the median stride after the previous kept one (re-contacts)."""
    if len(hs) < 4:
        return hs
    med = np.percentile(np.diff(hs), 75)           # re-contacts shorten many intervals: not the median
    keep = [hs[0]]
    for i in hs[1:]:
        if i - keep[-1] >= frac * med:
            keep.append(i)
    return np.asarray(keep, int)


def _window(rec, t0: float, t1: float):
    t = rec["t"]
    return np.flatnonzero((t >= t0) & (t <= t1))


def gait_cycles(rec, t0: float = 0.0, t1: float = np.inf, key: str = "q") -> List[np.ndarray]:
    """Joint-angle curves (N_PHASE x channels) of each right-heel-strike to right-heel-strike cycle in [t0, t1];
    key "q": the human's joints, "q_exo": the exo encoders."""
    hs = heel_strikes(rec, 1)
    hs = hs[(rec["t"][hs] >= t0) & (rec["t"][hs] <= t1)]
    out = []
    for a, b in zip(hs[:-1], hs[1:]):
        ph = np.linspace(0.0, 1.0, b - a + 1)
        out.append(np.stack([np.interp(np.linspace(0, 1, N_PHASE), ph, rec[key][a:b + 1, c])
                             for c in range(rec[key].shape[1])], axis=1))
    return out


def step_times(rec, t0: float = 0.0, t1: float = np.inf) -> np.ndarray:
    hs = np.sort(np.concatenate([heel_strikes(rec, 0), heel_strikes(rec, 1)]))
    hs = hs[(rec["t"][hs] >= t0) & (rec["t"][hs] <= t1)]
    return np.diff(rec["t"][hs])


@dataclass
class GaitBaseline:
    """The person's fresh gait: mean and SD joint-angle curves over the cycle, speed and step-time statistics."""
    mean: np.ndarray
    sd: np.ndarray
    chan: Sequence[str]
    speed: float
    step_time: float
    step_time_cv: float
    n_cycles: int

    @classmethod
    def from_records(cls, recs, t0: float = 0.0, t1: float = np.inf, key: str = "q") -> "GaitBaseline":
        cyc = [c for r in recs for c in gait_cycles(r, t0, t1, key)]
        st = np.concatenate([step_times(r, t0, t1) for r in recs])
        sp = np.mean([np.mean(r["com_vx"][_window(r, t0, t1)]) for r in recs])
        C = np.stack(cyc)
        chan = recs[0]["chan"] if key == "q" else ["hip_l", "knee_l", "hip_r", "knee_r"]
        return cls(C.mean(0), C.std(0), chan, float(sp), float(st.mean()), float(st.std() / st.mean()), len(cyc))


ACTUATED = ("hip", "knee")         # joints the Indego drives


def gait_deviation_scores(chan, ratio) -> Dict[str, float]:
    """Worst per-joint deviation ratio over the actuated joints (pass/fail), the ankles, and all joints."""
    ratio = np.asarray(ratio, float)
    act = np.array([c.startswith(ACTUATED) for c in chan])
    return {"gait_deviation": float(np.max(ratio[act])), "gait_deviation_ankle": float(np.max(ratio[~act])),
            "gait_deviation_all": float(np.max(ratio))}


def _positive_work(power, dt):
    return float(np.sum(np.clip(power, 0.0, None)) * dt)


def evaluate(rec, baseline: GaitBaseline = None, t0: float = 0.0, t1: float = np.inf) -> Dict[str, float]:
    """Success-criteria metrics of one rollout over [t0, t1] (s since walking began to be recorded)."""
    w = _window(rec, t0, min(t1, rec["t"][-1]))
    dt, T = rec["dt"], max(len(w) * rec["dt"], 1e-9)
    out = {"fell": bool(rec["fell"]), "duration": float(rec["t"][-1])}
    out["speed"] = float(np.mean(rec["com_vx"][w]))
    st = step_times(rec, t0, t1)
    out["step_time"] = float(st.mean()) if len(st) else np.nan
    out["step_time_cv"] = float(st.std() / st.mean()) if len(st) > 2 else np.nan
    # hip + knee channels of the human (hip_flexion, knee_angle on each side) and of the exo (hip_l, knee_l, ...)
    hk = [i for i, c in enumerate(rec["chan"]) if c.startswith(("hip", "knee"))]
    p_h = rec["tau_h"][w][:, hk] * rec["qd"][w][:, hk]
    p_e = rec["tau_exo"][w] * rec["qd_exo"][w]
    W_h, W_e = _positive_work(p_h, dt), _positive_work(p_e, dt)
    out["human_pos_work_per_s"] = W_h / T
    out["exo_pos_work_per_s"] = W_e / T
    out["human_share"] = W_h / (W_h + W_e) if W_h + W_e > 0 else np.nan
    out["exo_tau_sq"] = float(np.mean(np.sum(rec["tau_exo"][w] ** 2, axis=1)))
    out["drive"] = float(np.mean(rec["drive"][w]))
    out["fatigue"] = float(rec["fatigue"][w][-1]) if len(w) else 0.0
    if baseline is not None:
        cyc = gait_cycles(rec, t0, t1)
        out["n_cycles"] = len(cyc)
        if cyc:
            mean = np.stack(cyc).mean(0)
            dev = np.sqrt(np.mean((mean - baseline.mean) ** 2, axis=0))
            var = np.sqrt(np.mean(baseline.sd ** 2, axis=0))
            ratio = dev / np.maximum(var, np.radians(0.5))
            for c, r_ in zip(rec["chan"], ratio):
                out[f"dev_{c}"] = float(r_)
            out.update(gait_deviation_scores(rec["chan"], ratio))
        else:
            out.update(gait_deviation=np.nan, gait_deviation_ankle=np.nan, gait_deviation_all=np.nan)
    return out
