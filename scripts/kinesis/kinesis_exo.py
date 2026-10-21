"""Fold the Indego into KINESIS's MyoLegs model (in place): exo mass, centre of mass and inertia added to the human
segments they move with, drive inertia / friction / damping added at the hip and knee joints
(indego_exo_params.json, from exo_params.py). No bodies are added, so KINESIS's observation keeps its size and the
pretrained policy can continue training on the model. Scales allow randomising the exo per episode.

Run with the KINESIS environment (it only needs mujoco and numpy).
"""
import json
import os

import mujoco
import numpy as np

PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indego_exo_params.json")
AFO_NEUTRAL_DEG = 5.0     # indego_sim's AFO spring neutral: 5 deg dorsiflexion (ExoParams.afo_neutral_deg)
AFO_DAMPING = 2.0         # N*m*s/rad (ExoParams.afo_damping)


def _quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def _mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def add_exo(m, data=None, params=None, mass_scale=1.0, friction_scale=1.0, inertia_scale=None,
            k_hip_abduction=0.0, k_hip_rotation=0.0, k_ankle=0.0, ankle_neutral_deg=AFO_NEUTRAL_DEG):
    """Add the exo to MjModel `m` (MuJoCo model of KINESIS's MyoLegs) in place; returns the added mass (kg).
    mass_scale scales the added masses and inertias, friction_scale the joint friction and damping; inertia_scale
    (default = mass_scale) scales the drives' joint inertia. k_* (N*m/rad) add the exo's constraint of the person's
    other hip motions and ankle as springs: the Indego is a sagittal-plane exo whose frontal hip joints and AFO resist
    hip abduction, hip rotation and ankle motion (through the cuffs). The hip springs act about the model's reference
    posture, the ankle spring (the AFO, with its damping) about ankle_neutral_deg of dorsiflexion as in indego_sim -
    about 0 deg it would push the foot into plantarflexion all through stance (~13 N*m at 150 N*m/rad)."""
    p = params or json.load(open(PARAMS))
    inertia_scale = mass_scale if inertia_scale is None else inertia_scale
    added = 0.0
    for body, seg in p["segments"].items():
        b = m.body(body).id
        m0, c0 = float(m.body_mass[b]), m.body_ipos[b].copy()
        R0 = _quat_to_mat(m.body_iquat[b])
        I0 = R0 @ np.diag(m.body_inertia[b]) @ R0.T                       # body-frame axes, about its own COM
        me, ce, Ie = mass_scale * seg["mass"], np.asarray(seg["com"]), mass_scale * np.asarray(seg["inertia"])
        mt = m0 + me
        ct = (m0 * c0 + me * ce) / mt
        I = np.zeros((3, 3))
        for mi, ci, Ii in ((m0, c0, I0), (me, ce, Ie)):
            r = ci - ct
            I += Ii + mi * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        w, V = np.linalg.eigh(I)
        if np.linalg.det(V) < 0:
            V[:, 0] *= -1
        m.body_mass[b], m.body_ipos[b] = mt, ct
        m.body_inertia[b], m.body_iquat[b] = w, _mat_to_quat(V)
        added += me
    for joint, jp in p["joints"].items():
        da = m.jnt_dofadr[m.joint(joint).id]
        m.dof_armature[da] += inertia_scale * jp["armature"]
        m.dof_frictionloss[da] += friction_scale * jp["frictionloss"]
        m.dof_damping[da] += friction_scale * jp["damping"]
    for joints, k in ((("hip_adduction_l", "hip_adduction_r"), k_hip_abduction),
                      (("hip_rotation_l", "hip_rotation_r"), k_hip_rotation),
                      (("ankle_angle_l", "ankle_angle_r"), k_ankle)):
        for j in joints:
            m.jnt_stiffness[m.joint(j).id] += k                          # about qpos_spring (reference posture)
    if k_ankle > 0:          # MyoLeg's ankle has no joint spring of its own, so its spring reference is the AFO's
        for j in ("ankle_angle_l", "ankle_angle_r"):                     # + = dorsiflexion in MyoLeg
            jid = m.joint(j).id
            m.qpos_spring[m.jnt_qposadr[jid]] = np.radians(ankle_neutral_deg)
            m.dof_damping[m.jnt_dofadr[jid]] += AFO_DAMPING
    mujoco.mj_setConst(m, data if data is not None else mujoco.MjData(m))
    return added
