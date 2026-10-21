"""The Indego as seen by the leg model it is strapped to: each exo part's mass, centre of mass and inertia folded into
the human segment it moves with (rigid attachment, indego_sim's standing pose), plus the drives' joint-side inertia
and friction - for fine-tuning KINESIS with the exo (scripts/kinesis/kinesis_exo.py applies it to KINESIS's model).

Run with indego_sim's environment:  python scripts/kinesis/exo_params.py -> scripts/kinesis/indego_exo_params.json
"""
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

# exo body -> human body it moves with
ATTACH = {"exo_hipunit": "pelvis",
          "exo_thigh_{s}": "femur_{s}", "exo_thigh_cuff_{s}": "femur_{s}",
          "exo_shank_{s}": "tibia_{s}", "exo_shank_cuffP_{s}": "tibia_{s}", "exo_shank_cuffD_{s}": "tibia_{s}",
          "exo_afo_{s}": "calcn_{s}", "exo_shoe_{s}": "calcn_{s}"}
JOINTS = {"exo_hip_{s}": "hip_flexion_{s}", "exo_knee_{s}": "knee_angle_{s}"}


def main():
    from indego_sim.config import ExoParams, SimConfig
    from indego_sim.human.sci import get_preset
    from indego_sim.model.builder import build_model
    m, info = build_model(subject=get_preset("healthy"), sim=SimConfig(planar=False), with_exo=True)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_kinematics(m, d)
    groups = {}
    for exo_pat, hum_pat in ATTACH.items():
        for s in ("l", "r"):
            e, h = exo_pat.format(s=s), hum_pat.format(s=s)
            groups.setdefault(h, set()).add(e)
    out = {"segments": {}, "joints": {}}
    for h, exos in groups.items():
        hid = m.body(h).id
        Rh, ph = d.xmat[hid].reshape(3, 3), d.xpos[hid]
        mass, mc, parts = 0.0, np.zeros(3), []
        for e in sorted(exos):
            eid = m.body(e).id
            me = float(m.body_mass[eid])
            c = Rh.T @ (d.xipos[eid] - ph)                                 # COM in the human body frame
            R = Rh.T @ d.ximat[eid].reshape(3, 3)
            Ie = R @ np.diag(m.body_inertia[eid]) @ R.T                     # about its own COM, human-frame axes
            parts.append((me, c, Ie))
            mass += me
            mc += me * c
        com = mc / mass
        I = np.zeros((3, 3))
        for me, c, Ie in parts:                                            # parallel axis to the combined COM
            r = c - com
            I += Ie + me * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        out["segments"][h] = {"mass": mass, "com": com.tolist(), "inertia": I.tolist(), "parts": sorted(exos)}
    p = ExoParams()
    for exo_pat, hum_pat in JOINTS.items():
        for s in ("l", "r"):
            jid = m.joint(exo_pat.format(s=s)).id
            da = m.jnt_dofadr[jid]
            motor = p.hip_motor if "hip" in exo_pat else p.knee_motor
            out["joints"][hum_pat.format(s=s)] = {
                "armature": float(m.dof_armature[da]) + motor.rotor_inertia * motor.gear ** 2,
                "frictionloss": float(m.dof_frictionloss[da]), "damping": float(m.dof_damping[da])}
    out["total_mass"] = sum(v["mass"] for v in out["segments"].values())
    out["note"] = ("indego_sim healthy subject, standing keyframe; exo parts rigidly attached; joint values are the exo "
                   "joint's armature + reflected rotor inertia, Coulomb friction and damping (zero-torque drives)")
    path = os.path.join(HERE, "indego_exo_params.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"exo mass {out['total_mass']:.2f} kg -> {path}")
    for h, v in out["segments"].items():
        print(f"  {h:8s} +{v['mass']:.2f} kg at {np.round(v['com'], 3)}")
    print("  joints:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out["joints"].items()})


if __name__ == "__main__":
    main()
