"""Phase 4: the exo teacher's training environment (Gymnasium API).

The person (robust T12 AIS D walker, sagittal plane, no support) walks in the Indego while their volitional drive and
muscle fatigue degrade within the episode (indego_aan.deficits); the policy commands hip and knee exo torques.

* observation (teacher, sim only; plan "Exo policy specification"): the Indego sensors (ExoSensors.flatten: encoders,
  firmware velocity, motor-current torque estimate, thigh / hip / trunk IMUs, 44 values) + the true deficit (drive and
  mean fatigue per muscle group) + human joint torques + gait phase (from the true heel strikes) + the human joint
  angles and their error to the person's fresh curve at that phase.
* action: hip and knee torques in [-1, 1] x `tau_scale`, held for `hold` control periods (100 Hz decisions). With
  base="paan" the action is a correction added to the personal assist-as-needed controller
  (indego_aan.controllers.PersonalAAN, run at 200 Hz; residual policy learning), whose torque and gains are then
  also observed; the total is clipped to `tau_scale`.
* reward: r = w_g r_gait + w_s r_stab - w_a |tau|^2 + w_p r_part - w_j |dtau|^2 (plan), each term normalised to ~[0, 1]:
    |tau|^2 mean squared total exo torque / tau_scale^2, divided by max(1 - capacity, 1 / torque_fresh_scale) -
            assistance costs up to `torque_fresh_scale` times more while the person can still do the work (the
            criterion: exo torque rises only when the human's contribution falls)
    r_gait  exp(-mean((max(|q - q_fresh(phase)| - gait_band x sd_fresh(phase), 0) / gait_sigma)^2) / 2) over hip,
            knee, ankle of both legs - a tracking stand-in for the plan's AMP discriminator (the reference is the
            person's own fresh gait; phase from heel strikes, so a slower stride with a normal pattern is not
            penalised). Deviations within `gait_band` fresh SDs cost nothing: the criterion is "within the fresh
            variability", and rewarding tighter tracking than that pays the policy to push a fresh person.
    r_stab  1 per step alive - the sagittal margin-of-stability shortfall: the extrapolated COM (Hof) further than
            `mos_floor_ant` ahead of the front toe of the loaded feet (beyond what the fresh gait does in single
            support; per 10 cm), or behind the rear heel (per 5 cm); falls terminate the episode with `fall_penalty`
    r_part  -(shortfall of the human's share of hip + knee positive work below the capacity the deficit leaves,
            drive x (1 - hip/knee fatigue))  (2-s running averages)
"""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym, spaces = None, None

from .deficits import Deficits, DriveProcess, Fatigue3CCr
from .scenario import build_walking_model, make_walking_sim

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "reference_sci_t12_aisD_robust.npz")
FATIGUE_GROUPS = ("hip_extensors", "hip_flexors", "knee_extensors", "knee_flexors", "ankle_plantarflexors",
                  "ankle_dorsiflexors")
HIP_KNEE_GROUPS = ("hip_extensors", "hip_flexors", "knee_extensors", "knee_flexors")


@dataclass
class RewardWeights:
    w_g: float = 1.0
    w_s: float = 1.0
    w_a: float = 1.0
    torque_fresh_scale: float = 10.0  # torque cost multiplier at full capacity (1: no deficit scaling)
    w_p: float = 1.0                  # participation: the therapist's dial (challenge vs support)
    w_j: float = 0.1
    gait_sigma_deg: float = 5.0
    gait_band: float = 2.0            # fresh SDs of free deviation (as the personal AAN's dead band)
    mos_floor_ant: float = -0.20      # m: the fresh gait's extrapolated COM passes the front toe by up to ~0.2 m
    fall_penalty: float = 10.0


@dataclass
class DeficitRanges:
    """Per-episode deficit draw (plan "Randomization": drive 0.3-1.0 drifting with lapses; fatigue per group)."""
    p_fresh: float = 0.15
    p_drive: float = 0.6
    drive_level: tuple = (0.3, 1.0)
    drive_drop_at: tuple = (1.0, 8.0)          # s after the walk starts
    drive_ramp: tuple = (1.0, 5.0)             # s
    drift_sd: tuple = (0.0, 0.1)
    lapse_rate: tuple = (0.0, 0.1)             # 1/s
    p_fatigue: float = 0.6
    fatigue_rate: tuple = (3.0, 100.0)         # x the sustained-contraction rates, log-uniform
    group_scale: tuple = (0.5, 2.0)
    p_prefatigue: float = 0.5
    prefatigue: tuple = (0.0, 0.5)             # MF of the hip and knee extensors at the start


class AANEnv(gym.Env if gym else object):
    def __init__(self, episode_s: float = 20.0, hold: int = 2, tau_scale=(40.0, 40.0, 40.0, 40.0),
                 weights: RewardWeights = None, ranges: DeficitRanges = None, deficit: dict = None,
                 reference: str = REF, seed: int = 0, base: str = None, base_params: dict = None):
        """deficit: fixed deficit for evaluation, e.g. {"kind": "drive", "level": 0.4} (see draw_deficit); None
        draws one per episode from `ranges`. base: None (the policy commands the torque) or "paan" (the policy adds
        a correction to the personal AAN, PersonalAANParams(**base_params), which is told the true capacity - used
        only if base_params sets deficit_gain)."""
        self.episode_s, self.hold = episode_s, hold
        self.tau_scale = np.asarray(tau_scale, float)
        self.w, self.ranges, self.fixed = weights or RewardWeights(), ranges or DeficitRanges(), deficit
        ref = np.load(reference)
        self.mean_h, self.sd_h = ref["mean_h"], ref["sd_h"]
        self.stride0 = 2.0 * float(ref["step_time"])
        self.base, self.base_params = base, dict(base_params or {})
        self.paan_ref = {k: ref[k] for k in ("mean", "sd", "step_time")}
        self.ph_grid = np.linspace(0.0, 1.0, self.mean_h.shape[0])
        self.rng = np.random.default_rng(seed)
        self.model = None
        self.sim = None
        self.recorder = None                    # optional callable after every control step (evaluation)
        n_obs = 44 + 1 + len(FATIGUE_GROUPS) + 6 + 3 + 12 + (8 if base else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    # ---------------------------------------------------------------------------------------------- deficits
    def draw_deficit(self):
        r, g = self.ranges, self.rng
        if self.fixed is not None:
            return dict(self.fixed)
        if g.random() < r.p_fresh:
            return {"kind": "fresh"}
        out = {"kind": "random"}
        if g.random() < r.p_drive:
            t0, ramp = g.uniform(*r.drive_drop_at), g.uniform(*r.drive_ramp)
            out["drive"] = dict(level=g.uniform(*r.drive_level), t0=t0, ramp=ramp, drift_sd=g.uniform(*r.drift_sd),
                                lapse_rate=g.uniform(*r.lapse_rate))
        if g.random() < r.p_fatigue:
            out["fatigue"] = dict(rate=float(np.exp(g.uniform(*np.log(r.fatigue_rate)))),
                                  group_scale={k: g.uniform(*r.group_scale) for k in FATIGUE_GROUPS},
                                  pre=g.uniform(*r.prefatigue) if g.random() < r.p_prefatigue else 0.0)
        return out

    def _attach(self, spec, walk_t0):
        sim, seed = self.sim, int(self.rng.integers(1 << 31))
        drive = fat = None
        kind = spec["kind"]
        if kind == "drive":                                      # evaluation: drop to `level` over 10-15 s
            drive = DriveProcess(schedule=[(0.0, 1.0), (10.0, 1.0), (15.0, spec["level"]), (1e9, spec["level"])])
        elif kind == "fatigue":
            fat = Fatigue3CCr(sim.human.mm.names, rate_scale=spec["level"])
        elif kind == "prefatigue":
            fat = Fatigue3CCr(sim.human.mm.names)
            fat.reset(np.array([spec["level"] if g in ("hip_extensors", "knee_extensors") else 0.0 for g in fat.group]))
        elif kind == "random":
            if "drive" in spec:
                p = spec["drive"]
                t0 = walk_t0 + p["t0"]
                drive = DriveProcess(schedule=[(0.0, 1.0), (t0, 1.0), (t0 + p["ramp"], p["level"]),
                                               (1e9, p["level"])], drift_sd=p["drift_sd"], lapse_rate=p["lapse_rate"],
                                     seed=seed)
            if "fatigue" in spec:
                p = spec["fatigue"]
                fat = Fatigue3CCr(sim.human.mm.names, rate_scale=p["rate"], group_scale=p["group_scale"])
                fat.reset(np.array([p["pre"] if g in ("hip_extensors", "knee_extensors") else 0.0 for g in fat.group]))
        return Deficits(sim, drive=drive, fatigue=fat)

    # ---------------------------------------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        """options: {"sim_seed": s} runs the person's seed s (as make_walking_sim(seed=s)) for evaluation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        sim_seed = (options or {}).get("sim_seed", int(self.rng.integers(1 << 31)))
        if self.sim is not None:
            # free the last episode now: the sim's objects form reference cycles that Python's occasional full
            # collections free only late, so a long-running worker would hold ~10 dead sims (~50 MB each)
            self.defi.detach()
            self.sim = self.defi = self.recorder = None
            gc.collect()
        if self.model is None:
            self.model = build_walking_model()        # pristine; each sim works on its own copy
        self.sim, obs = make_walking_sim(seed=sim_seed, model=self.model)
        sim, m, hc = self.sim, self.sim.m, self.sim.human
        self.spec = self.draw_deficit()
        self.defi = self._attach(self.spec, 0.0)
        chan = [f"{j}_{s}" for s in ("l", "r") for j in ("hip_flexion", "knee_angle", "ankle_angle")]
        idx = [hc.dof_names.index(c) for c in chan]
        self.q_adr, self.d_adr = hc.qadr[idx], hc.dadr[idx]
        self.mm_rows = [hc.mm.dof_names.index(c) for c in chan]
        self.hk = [0, 1, 3, 4]                                       # hip and knee among chan
        self.exo_dof = np.array([m.jnt_dofadr[m.joint(j).id] for j in sim.info.exo_joints])
        self.shoe = {s: (m.geom(f"foot_shoe_{s}").id, m.geom(f"bofoot_shoe_{s}").id) for s in ("l", "r")}
        self.leg_len = float(sim.d.subtree_com[sim.root][2])
        self.t_hs, self.stride = None, self.stride0
        self.loaded_prev = False
        self.p_h = self.p_e = 0.0
        self.mos_ant, self.mos_post = 0.0, 0.2
        self.tau_prev = np.zeros(4)
        self.tau_b = np.zeros(4)
        self.tau = np.zeros(4)
        self.cap = 1.0
        self.base_ctrl = None
        if self.base == "paan":
            from .controllers import PersonalAAN
            from .controllers import PersonalAANParams
            bp = dict(self.base_params)
            if "k_max" in bp:
                bp["k_max"] = tuple(bp["k_max"])
            self.base_ctrl = PersonalAAN(self.paan_ref, PersonalAANParams(**bp), dt=sim.ctrl_dt)
        elif self.base is not None:
            raise ValueError(f"unknown base controller {self.base!r}")
        self.tau_h = hc.mm.moment_arms(sim.d)[self.mm_rows] @ sim.d.actuator_force[hc.mm.ids]
        self.t = 0.0
        self._obs_raw = obs
        return self._observation(), {"deficit": self.spec}

    def _phase(self):
        if self.t_hs is None:
            return 0.0
        return min((self.t - self.t_hs) / self.stride, 1.0)

    def _q_ref(self, ph):
        return np.array([np.interp(ph, self.ph_grid, self.mean_h[:, c]) for c in range(6)])

    def _sd_ref(self, ph):
        return np.array([np.interp(ph, self.ph_grid, self.sd_h[:, c]) for c in range(6)])

    def _observation(self):
        sim, d = self.sim, self.sim.d
        sens = sim.sensors.flatten(self._obs_raw)
        fat = self.defi.fatigue.group_mean() if self.defi.fatigue is not None else {}
        f = [fat.get(g, 0.0) for g in FATIGUE_GROUPS]
        tau_h = self.tau_h
        ph = self._phase()
        q = d.qpos[self.q_adr]
        x = np.concatenate([sens, [self.defi.drive_now], f, tau_h / 100.0,
                            [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph), float(self.t_hs is not None)],
                            q, self._q_ref(ph) - q])
        if self.base_ctrl is not None:
            x = np.concatenate([x, self.tau_b / self.tau_scale, self.base_ctrl.k])
        return x.astype(np.float32)

    def _foot_x(self, s):
        d = self.sim.d
        out = []
        for gid, sign in zip(self.shoe[s], (-1.0, 1.0)):
            R = d.geom_xmat[gid].reshape(3, 3)
            ext = float(np.abs(R[0]) @ self.sim.m.geom_size[gid])
            out.append(d.geom_xpos[gid][0] + sign * ext)
        return out                                                   # heel x, toe x

    def _control_step(self, tau_act):
        sim, d, w = self.sim, self.sim.d, self.w
        com_prev = float(d.subtree_com[sim.root][0])
        if self.base_ctrl is not None:
            self.base_ctrl.capacity = self.cap
            self.tau_b = np.asarray(self.base_ctrl(self._obs_raw), float)
        tau = self.tau = np.clip(self.tau_b + tau_act, -self.tau_scale, self.tau_scale)
        self._obs_raw = sim.step(tau, record=False)
        if self.recorder is not None:
            self.recorder()
        dt = sim.ctrl_dt
        self.t += dt
        # true right heel strike -> gait phase
        loaded = {s: float(sim._grf[s][2]) > 0.05 * sim.BW for s in ("l", "r")}
        if loaded["r"] and not self.loaded_prev and (self.t_hs is None or self.t - self.t_hs > 0.3):
            if self.t_hs is not None:
                self.stride = float(np.clip(0.7 * self.stride + 0.3 * (self.t - self.t_hs), 0.5 * self.stride0,
                                            3.0 * self.stride0))
            self.t_hs = self.t
        self.loaded_prev = loaded["r"]
        # gait
        q = d.qpos[self.q_adr]
        ph = self._phase()
        dev = np.maximum(np.abs(q - self._q_ref(ph)) - w.gait_band * self._sd_ref(ph), 0.0)
        err = np.degrees(dev) / w.gait_sigma_deg
        r_gait = float(np.exp(-0.5 * np.mean(err ** 2))) if self.t_hs is not None else 0.0
        # stability: sagittal extrapolated COM inside the loaded feet
        com = float(d.subtree_com[sim.root][0])
        xcom = com + (com - com_prev) / dt / np.sqrt(9.81 / self.leg_len)
        feet = [self._foot_x(s) for s in ("l", "r") if loaded[s]]
        if feet:
            self.mos_ant = max(f[1] for f in feet) - xcom
            self.mos_post = xcom - min(f[0] for f in feet)
        r_stab = (1.0 - float(np.clip((w.mos_floor_ant - self.mos_ant) / 0.10, 0.0, 1.0))
                  - float(np.clip(-self.mos_post / 0.05, 0.0, 1.0)))
        # participation: running positive work of the human (hip, knee) vs the exo
        tau_h = self.tau_h = sim.human.mm.moment_arms(d)[self.mm_rows] @ d.actuator_force[sim.human.mm.ids]
        ph_ = float(np.sum(np.clip(tau_h[self.hk] * d.qvel[self.d_adr][self.hk], 0.0, None)))
        pe_ = float(np.sum(np.clip(d.qfrc_actuator[self.exo_dof] * d.qvel[self.exo_dof], 0.0, None)))
        a = dt / 2.0
        self.p_h += a * (ph_ - self.p_h)
        self.p_e += a * (pe_ - self.p_e)
        share = self.p_h / (self.p_h + self.p_e) if self.p_h + self.p_e > 1e-6 else 1.0
        fat = self.defi.fatigue.group_mean() if self.defi.fatigue is not None else {}
        cap = self.cap = self.defi.drive_now * (1.0 - np.mean([fat.get(g, 0.0) for g in HIP_KNEE_GROUPS]))
        r_part = -max(cap - share, 0.0)
        torque = float(np.mean((tau / self.tau_scale) ** 2)) / max(1.0 - cap, 1.0 / w.torque_fresh_scale)
        return dict(r_gait=r_gait, r_stab=r_stab, r_part=r_part, torque=torque, mos_ant=self.mos_ant,
                    mos_post=self.mos_post, share=share, cap=cap)

    def step(self, action):
        w = self.w
        tau_act = np.clip(np.asarray(action, float), -1.0, 1.0) * self.tau_scale
        terms = {}
        for _ in range(self.hold):
            k = self._control_step(tau_act)
            for key, v in k.items():
                terms[key] = terms.get(key, 0.0) + v / self.hold
            if self.sim.fell:
                break
        tau = self.tau
        du = (tau - self.tau_prev) / self.tau_scale
        self.tau_prev = tau
        r = (w.w_g * terms["r_gait"] + w.w_s * terms["r_stab"] - w.w_a * terms["torque"]
             + w.w_p * terms["r_part"] - w.w_j * float(np.mean(du ** 2)))
        terminated = bool(self.sim.fell)
        if terminated:
            r -= w.fall_penalty
        truncated = self.t >= self.episode_s
        info = dict(terms, drive=self.defi.drive_now, tau=tau, tau_base=self.tau_b.copy(), fell=terminated)
        return self._observation(), float(r), terminated, truncated, info


def env_from_config(cfg: dict = None, **kw) -> AANEnv:
    """AANEnv from a run's env.json ({"base": ..., "weights": {...}, "ranges": {...}}) plus keyword overrides."""
    cfg = dict(cfg or {})
    if "weights" in cfg:
        cfg["weights"] = RewardWeights(**cfg["weights"])
    if "ranges" in cfg:
        cfg["ranges"] = DeficitRanges(**cfg["ranges"])
    cfg.update(kw)
    return AANEnv(**cfg)
