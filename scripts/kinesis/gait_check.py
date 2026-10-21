# Uses KINESIS (github.com/amathislab/Kinesis), Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI,
# EPFL, BSD-3-Clause: the policy is loaded and run as in its src/agents/agent_im.py (eval_single_thread).
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# copyright notices, this list of conditions and the disclaimer are retained. THIS SOFTWARE IS PROVIDED "AS IS",
# WITHOUT WARRANTY OF ANY KIND.
"""Gait check of a KINESIS legs policy in KINESIS's own simulation: how the pelvis and hips move, which KINESIS's
evaluation (positions of pelvis, knees, ankles and toes) does not see. Each reference is walked for <seconds>
(evaluation mode, no reset on deviation); after the first 3 s it reports the ranges (2nd-98th percentile) of the
pelvis heading relative to the walking direction (yaw), its pitch and roll, the same for the reference pelvis, the
hip rotation / adduction / flexion and knee ranges, and when the evaluation would have ended (mean distance of the
tracked bodies > 0.5 m). A human-like walk at these speeds: pelvis yaw ~10-15 deg, pitch ~3-5, roll ~5-8 deg; hip
rotation ~10-15 deg; the pretrained KINESIS policy twists the pelvis by ~80 deg.

Run with the KINESIS environment, from the KINESIS checkout:

    cd <kinesis>; python <this file> <model_dir> <moe|lattice> <seconds> <motion.pkl> <pose.pkl> [<motion> <pose> ...]
        [exo / deficit options of exo_kinesis.py, e.g. --exo-mode fixed --exo-kabd 0 --exo-krot 0 --exo-kank 150]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.getcwd())
sys.path.append(HERE)

import numpy as np  # noqa: E402

JOINTS = ("hip_rotation_r", "hip_rotation_l", "hip_adduction_r", "hip_adduction_l", "hip_flexion_r", "hip_flexion_l",
          "knee_angle_r", "knee_angle_l")


def quat_wxyz_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_agent(model_dir, actor, motion, pose):
    import torch
    from hydra import compose, initialize_config_dir
    from src.agents import agent_dict
    with initialize_config_dir(config_dir=os.path.join(os.getcwd(), "cfg"), version_base=None):
        cfg = compose(config_name="config_legs", overrides=[
            "exp_name=gait_check", "epoch=-1", f"learning.actor_type={actor}", "run=eval_run_legs", "run.headless=True",
            f"run.motion_file={motion}", f"run.initial_pose_file={pose}", "env.termination_distance=0.5", "no_log=True",
            "run.num_motions=1"])
    cfg.output_dir = model_dir
    torch.set_default_dtype(torch.float32)
    return agent_dict[cfg.learning.agent_name](cfg, torch.float32, torch.device("cpu"), training=True,
                                                checkpoint_epoch=cfg.epoch)


def walk(agent, seconds):
    """Per decision: time, pelvis rotation (3x3), pelvis position, reference pelvis rotation, joint angles, distance."""
    import torch
    from src.learning.learning_utils import to_test
    from src.learning.policy_moe import PolicyMOE
    env = agent.env
    m, d = env.mj_model, env.mj_data
    to_test(*agent.sample_modules)
    env.start_eval(im_eval=True)
    next(iter(env.forward_motions()))
    obs, _ = env.reset()
    env.disable_reset = True
    state = agent.preprocess_obs(obs)
    pel = m.body("pelvis").id
    qa = [m.jnt_qposadr[m.joint(j).id] for j in JOINTS]
    rows = []
    for _ in range(int(seconds * 30)):
        with torch.no_grad():
            x = torch.from_numpy(state).to(torch.float32)
            if isinstance(agent.policy_net, PolicyMOE):
                a = agent.policy_net.select_action(x, True)[0][0].numpy()
            else:
                a = agent.policy_net.select_action(x, True)[0].numpy()
        obs, *_ = env.step(agent.preprocess_actions(a))
        state = agent.preprocess_obs(obs)
        t = env.cur_t * env.dt + env._motion_start_times + env._motion_start_times_offset
        ref = env.get_state_from_motionlib_cache(env._sampled_motion_ids, t, env.global_offset)
        dist = np.linalg.norm(env.get_body_xpos()[env.reset_bodies_id] - ref.xpos[0, env.smpl_reset_ids], axis=-1)
        rows.append(np.concatenate([[float(t[0])], d.xmat[pel], d.xpos[pel], quat_wxyz_to_mat(ref.root_rot[0]).ravel(),
                                    np.degrees(d.qpos[qa]), [dist.mean()], ref.root_pos[0]]))
    return np.array(rows)


def direction(P):
    """Walking direction (unit, horizontal) of a path over ~1.4 s."""
    from scipy.ndimage import uniform_filter1d
    v = uniform_filter1d(np.gradient(P[:, :2], axis=0), 42, axis=0)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def metrics(rows):
    """Ranges from 3 s until the evaluation would have ended (a fall or loss of the reference), each pelvis against
    its own path; a walk that ends before 10 s is flagged (its ranges describe the fall, not the gait)."""
    t, Ms, P, Mr = rows[:, 0], rows[:, 1:10].reshape(-1, 3, 3), rows[:, 10:13], rows[:, 13:22].reshape(-1, 3, 3)
    J, dist, Pr = rows[:, 22:30], rows[:, 30], rows[:, 31:34]
    over = np.flatnonzero(dist > 0.5)
    t_end = float(t[over[0]]) if len(over) else None
    w = (t > 3.0) & (t < (t_end - 0.5 if t_end is not None else np.inf))
    out = {"would_end_s": t_end, "ended_early": t_end is not None and t_end < 10.0}
    if w.sum() < 30:
        w = t > -1                                                       # too short: whatever there is (flagged)
    rom = lambda x: float(np.ptp(np.percentile(x[w], [2, 98])))           # noqa: E731
    for name, M, fwd, f in (("pelvis", Ms, 0, direction(P)), ("ref_pelvis", Mr, 2, direction(Pr))):
        lat = np.stack([-f[:, 1], f[:, 0]], 1)                           # MyoLeg pelvis: x forward; SMPL: z forward
        h = np.arctan2(M[:, 1, fwd], M[:, 0, fwd]) - np.arctan2(f[:, 1], f[:, 0])
        yaw = np.degrees(np.angle(np.exp(1j * h)))
        u = M[:, :, 1]                                                      # both: y up
        out[f"{name}_yaw_rom"] = rom(yaw)
        out[f"{name}_pitch_rom"] = rom(np.degrees(np.arcsin(np.clip((u[:, :2] * f).sum(1), -1, 1))))
        out[f"{name}_roll_rom"] = rom(np.degrees(np.arcsin(np.clip((u[:, :2] * lat).sum(1), -1, 1))))
    for i, j in enumerate(JOINTS):
        out[f"{j}_rom"] = rom(J[:, i])
    return out


def main(argv):
    import exo_kinesis
    opts, rest = exo_kinesis.parse_exo_args(argv)
    if opts["mode"] != "none" or exo_kinesis.DEFICIT["ranges"]:
        exo_kinesis.install(opts)
    model_dir, actor, seconds = rest[1], rest[2], float(rest[3])
    pairs = list(zip(rest[4::2], rest[5::2]))
    keys = ("pelvis_yaw_rom", "pelvis_pitch_rom", "pelvis_roll_rom", "ref_pelvis_yaw_rom", "ref_pelvis_pitch_rom",
            "ref_pelvis_roll_rom", "hip_rotation_r_rom", "hip_adduction_r_rom", "hip_flexion_r_rom", "knee_angle_r_rom")
    allm = []
    for motion, pose in pairs:
        r = metrics(walk(load_agent(model_dir, actor, motion, pose), seconds))
        allm.append(r)
        print(f"GAIT {os.path.basename(motion)}: pelvis yaw/pitch/roll {r['pelvis_yaw_rom']:.1f}/{r['pelvis_pitch_rom']:.1f}/"
              f"{r['pelvis_roll_rom']:.1f} deg (reference {r['ref_pelvis_yaw_rom']:.1f}/{r['ref_pelvis_pitch_rom']:.1f}/"
              f"{r['ref_pelvis_roll_rom']:.1f}), hip rotation {r['hip_rotation_r_rom']:.1f}, adduction "
              f"{r['hip_adduction_r_rom']:.1f}, flexion {r['hip_flexion_r_rom']:.1f}, knee {r['knee_angle_r_rom']:.1f} deg"
              + (f"; ENDED EARLY at {r['would_end_s']:.1f} s (left out of the mean)" if r["ended_early"] else
                 f"; evaluation would end at {r['would_end_s']}"), flush=True)
    ok = [r for r in allm if not r["ended_early"]]
    if ok:
        mean = {k: float(np.mean([r[k] for r in ok])) for k in keys}
        print(f"GAIT MEAN ({len(ok)} of {len(allm)} walks): " + ", ".join(f"{k} {v:.1f}" for k, v in mean.items()),
              flush=True)


if __name__ == "__main__":
    main(sys.argv)
