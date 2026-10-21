"""A learned person in indego_sim: the pretrained KINESIS legs policy drives the 80 MyoLeg muscles.

The policy itself runs in KINESIS's own environment (scripts/kinesis/policy_server.py, another Python / MuJoCo
version); this side sends it the person's state every 1/30 s and turns the 80 target muscle lengths it returns into
activations with KINESIS's muscle PD law, at KINESIS's simulation rate (150 Hz):

    force = clip(kp (L_target - L) - kd dL/dt, -F_max, 0),  kp = 5 F_max, kd = 0.1 kp
    activation = clip((force - bias(L)) / min(-1, gain(L, dL/dt)), 0, 1)       (MuJoCo muscle gain and bias)

indego_sim's own human controller keeps running underneath for the trunk (lumbar motors) and its bookkeeping; only
the leg-muscle commands are replaced. No reflex layer yet. Optional deficits (low drive, fatigue) act on
the muscle commands (indego_aan.deficits.CommandDeficits).
"""
from __future__ import annotations

from multiprocessing.connection import Client

import mujoco
import numpy as np

AUTHKEY = b"indego_sim-kinesis"
LUMBAR = ("lumbar_flexion", "lumbar_bending", "lumbar_rotation")
TOUCH = (("r_foot", ("calcn_r", "talus_r")), ("r_toes", ("toes_r",)), ("l_foot", ("calcn_l", "talus_l")),
         ("l_toes", ("toes_l",)))


class KinesisHuman:
    def __init__(self, sim, port: int = 6010, decision_dt: float = 1.0 / 30.0, pd_dt: float = 1.0 / 150.0,
                 deficits=None):
        """deficits: optional indego_aan.deficits.CommandDeficits applied to the muscle commands."""
        self.sim, self.m, self.hc = sim, sim.m, sim.human
        self.deficits = deficits
        m = self.m
        self.conn = Client(("localhost", port), authkey=AUTHKEY)
        self.decision_dt, self.pd_dt = decision_dt, pd_dt
        legs = self._leg_bodies(m)
        self.joints = [(m.joint(i).name, m.jnt_qposadr[i], m.jnt_dofadr[i]) for i in range(m.njnt)
                       if m.jnt_type[i] in (2, 3) and m.joint(i).name not in LUMBAR
                       and m.body(m.jnt_bodyid[i]).name in legs]
        self.pelvis = m.body("pelvis").id
        self.ids = np.asarray(self.hc.mm.ids)
        self.F = m.actuator_biasprm[self.ids, 2].copy()                 # peak force per muscle
        self.touch_bodies = [{m.body(b).id for b in bodies} for _, bodies in TOUCH]
        self.target = None
        self.act = np.zeros(len(self.ids))
        self.ref_error = []
        self._orig = self.hc.step
        self.hc.step = self._step
        self.active = False

    @staticmethod
    def _leg_bodies(m):
        names = ("pelvis", "femur", "tibia", "talus", "calcn", "toes", "patella")
        return {m.body(i).name for i in range(m.nbody) if m.body(i).name.startswith(names)}

    # ------------------------------------------------------------------------------------------- state out
    def _state(self, d, t):
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_XBODY, self.pelvis, vel, 0)
        return {"cmd": "act", "t": float(t),
                "q": {n: float(d.qpos[qa]) for n, qa, _ in self.joints},
                "qd": {n: float(d.qvel[da]) for n, _, da in self.joints},
                "pelvis_pos": d.xpos[self.pelvis].tolist(), "pelvis_quat": d.xquat[self.pelvis].tolist(),
                "pelvis_angvel": vel[:3].tolist(), "pelvis_linvel": vel[3:].tolist(), "touch": self._touch(d)}

    def _touch(self, d):
        """Normal contact force on each foot segment (heel / forefoot, right then left), N."""
        out = [0.0] * len(TOUCH)
        f = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if 0 not in (b1, b2):                                       # foot-floor contacts only
                continue
            b = b2 if b1 == 0 else b1
            for k, bodies in enumerate(self.touch_bodies):
                if b in bodies:
                    mujoco.mj_contactForce(self.m, d, i, f)
                    out[k] += float(abs(f[0]))
        return out

    # ------------------------------------------------------------------------------------------- control
    def start(self):
        """Align the reference clip to the person's current pose and start following it."""
        d = self.sim.d
        self.conn.send({"cmd": "reset", "pelvis_pos": d.xpos[self.pelvis].tolist(),
                        "pelvis_quat": d.xquat[self.pelvis].tolist()})
        reply = self.conn.recv()
        self.t_start = d.time
        self.next_decision = self.next_pd = d.time
        self.active = True
        return reply

    def _pd_activation(self, d):
        m, ids = self.m, self.ids
        L, V = d.actuator_length[ids], d.actuator_velocity[ids]
        kp = 5.0 * self.F
        force = np.clip(kp * (self.target - L) - 0.1 * kp * V, -self.F, 0.0)
        act = np.empty(len(ids))
        for k, a in enumerate(ids):
            lr, acc0 = m.actuator_lengthrange[a], m.actuator_acc0[a]
            bias = mujoco.mju_muscleBias(L[k], lr, acc0, m.actuator_biasprm[a, :9])
            gain = min(-1.0, mujoco.mju_muscleGain(L[k], V[k], lr, acc0, m.actuator_gainprm[a, :9]))
            act[k] = (force[k] - bias) / gain
        return np.clip(act, 0.0, 1.0)

    def _step(self, d, *a, **k):
        it = self._orig(d, *a, **k)
        if not self.active:
            return it
        if d.time >= self.next_decision - 1e-9:
            self.conn.send(self._state(d, d.time - self.t_start))
            reply = self.conn.recv()
            self.target = np.asarray(reply["target_length"])
            self.ref_error.append((d.time - self.t_start, reply["ref_error"]))
            self.next_decision += self.decision_dt
        if d.time >= self.next_pd - 1e-9:
            self.act = self._pd_activation(d)
            self.next_pd += self.pd_dt
        d.ctrl[self.ids] = self.act if self.deficits is None else self.deficits.apply(d, self.act, self.hc.dt)
        return it

    def close(self):
        try:
            self.conn.send({"cmd": "close"})
        finally:
            self.conn.close()
            self.hc.step = self._orig
