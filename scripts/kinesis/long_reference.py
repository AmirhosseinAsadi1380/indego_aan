# The conversion to KINESIS's motion format (to_motion) is adapted from KINESIS (src/utils/convert_kit.py), Copyright
# (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL, BSD-3-Clause; KINESIS adapts PHC (Copyright (c)
# 2023 Carnegie Mellon University; Copyright (c) 2018-2023, NVIDIA Corporation), BSD-3-Clause. Redistribution and use in
# source and binary forms, with or without modification, are permitted provided that the copyright notices, this list
# of conditions and the disclaimer are retained (see github.com/amathislab/Kinesis/blob/main/LICENSE.md). THIS
# SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""Long walking references for KINESIS: KIT clips last <= ~10 s and start and end standing, Phase 1 needs 60 s of
walking at 0.5-1.0 m/s. For each target speed, the straight forward-walking clip whose steady speed is closest is
kept up to the start of one clean gait cycle (its own gait initiation), then that cycle is repeated until the
reference is long enough. The cycle is closed into a loop by spreading its small start/end mismatch (joint
rotations, root orientation including heading drift, height) linearly over the cycle, so there is no jump at the
seams; the forward displacement per cycle is kept. The result is converted exactly as KINESIS converts KIT.

Run with the KINESIS environment, from the KINESIS checkout:

    cd <kinesis>; .venv/Scripts/python <this file> <out_dir> [speeds=0.5,0.6,0.7,0.8,0.9,1.0] [seconds=65] [alternatives=1]

Writes <out_dir>/long_walk_<speed>.pkl (one clip each; with alternatives > 1 also the next-best candidates as
long_walk_<speed>_r<k>.pkl) and long_walk_index.json (source clip, cycle, speed). Then
make the initial poses with initial_poses.py.
"""
import json
import os
import sys

sys.path.append(os.getcwd())

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

FPS = 30


# ------------------------------------------------------------------------------------------------ clip analysis

def heading_speed(trans):
    """Horizontal speed [m/s] per frame (AMASS frame: z up)."""
    v = np.gradient(trans[:, :2], axis=0) * FPS
    return np.linalg.norm(v, axis=1)


def steady_window(trans, tol=0.15):
    """(start, end, speed) of the longest stretch whose speed stays within tol of the clip's mid-clip median."""
    sp = np.convolve(heading_speed(trans), np.ones(9) / 9, mode="same")
    n = len(sp)
    med = np.median(sp[n // 3: 2 * n // 3])
    ok = np.abs(sp / max(med, 1e-6) - 1) < tol
    best, cur = (0, 0), None
    for i, o in enumerate(np.r_[ok, False]):
        if o and cur is None:
            cur = i
        elif not o and cur is not None:
            if i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    return best[0], best[1], float(med)


def straightness(trans, a, b):
    """Net displacement / path length over [a, b): 1 = straight."""
    d = np.linalg.norm(trans[b - 1, :2] - trans[a, :2])
    p = np.sum(np.linalg.norm(np.diff(trans[a:b, :2], axis=0), axis=1))
    return d / max(p, 1e-6)


def pose_distance(pa, pb):
    """Summed rotation angle [rad] between two SMPL poses over the 21 body joints (root excluded)."""
    ra = R.from_rotvec(pa[3:66].reshape(-1, 3))
    rb = R.from_rotvec(pb[3:66].reshape(-1, 3))
    return float(np.sum((ra.inv() * rb).magnitude()))


def best_cycle(pose_aa, a, b, min_len=0.8, max_len=1.8):
    """(i0, i1): the gait cycle inside [a, b) whose end pose matches its start pose best (prefers the middle)."""
    lo, hi = int(min_len * FPS), int(max_len * FPS)
    mid = (a + b) / 2
    best, score = None, np.inf
    for i0 in range(a, b - lo):
        for i1 in range(i0 + lo, min(i0 + hi, b)):
            s = pose_distance(pose_aa[i0], pose_aa[i1]) + 0.002 * abs((i0 + i1) / 2 - mid)
            if s < score:
                best, score = (i0, i1), s
    return best, score


# ------------------------------------------------------------------------------------------------ the loop

def closed_cycle(pose_aa, trans, i0, i1):
    """Frames i0..i1-1 of the cycle, corrected so that frame i1 would coincide with frame i0 except for the forward
    displacement: per joint (and the root) the rotation mismatch D = R(i0) R(i1)^-1 is applied as D^s, s = 0..1
    over the cycle; translation increments are turned with the root's yaw correction, the height mismatch spread."""
    n = i1 - i0
    s = np.arange(n + 1) / n
    rots = R.from_rotvec(pose_aa[i0:i1 + 1, :66].reshape(n + 1, 22, 3).reshape(-1, 3)).as_rotvec().reshape(n + 1, 22, 3)
    out = rots.copy()
    yaw_corr = np.zeros(n + 1)
    for j in range(22):
        Rj = R.from_rotvec(rots[:, j])
        D = R.from_rotvec(rots[0, j]) * Rj[-1].inv()                   # rotation that maps the end onto the start
        corr = R.from_rotvec(np.outer(s, D.as_rotvec()))               # D^s
        if j == 0:
            out[:, j] = (corr * Rj).as_rotvec()                         # root: world-frame correction (left)
            yaw_corr = s * D.as_euler("ZYX")[0]
        else:
            out[:, j] = (corr * Rj).as_rotvec()                         # joints: same, in the parent frame
    # translation: rotate the horizontal increments with the yaw correction, spread the height mismatch
    d = np.diff(trans[i0:i1 + 1], axis=0)
    c, si = np.cos(yaw_corr[1:]), np.sin(yaw_corr[1:])
    dx = c * d[:, 0] - si * d[:, 1]
    dy = si * d[:, 0] + c * d[:, 1]
    t = np.zeros((n + 1, 3))
    t[1:, 0], t[1:, 1] = np.cumsum(dx), np.cumsum(dy)
    t[:, 2] = trans[i0:i1 + 1, 2] + s * (trans[i0, 2] - trans[i1, 2]) - trans[i0, 2]
    pose = np.zeros((n, 72))
    pose[:, :66] = out[:-1].reshape(n, 66)
    return pose, t[:-1], t[-1] * np.array([1, 1, 0])                 # frames, offsets, displacement per cycle


def build(pose_aa, trans, i0, i1, seconds):
    """Original frames up to i0 (gait initiation), then the closed cycle repeated."""
    cyc_pose, cyc_off, step = closed_cycle(pose_aa, trans, i0, i1)
    poses, ts = [pose_aa[:i0]], [trans[:i0]]
    n_total, k = i0, 0
    base = trans[i0].copy()
    while n_total < seconds * FPS:
        poses.append(cyc_pose)
        ts.append(base + k * step + cyc_off)
        n_total += len(cyc_pose)
        k += 1
    return np.concatenate(poses), np.concatenate(ts), k


# ------------------------------------------------------------------------------------------------ conversion

def to_motion(pose_aa, root_trans):
    """KINESIS's KIT conversion (src/utils/convert_kit.py) for one sequence at 30 fps."""
    from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
    from src.utils.smpl_skeleton.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
    N = pose_aa.shape[0]
    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
    pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
    pose_quat = R.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)
    skeleton_tree = SkeletonTree.from_mjcf("data/xml/smpl_humanoid.xml")
    root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]
    sk = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat),
                                                         root_trans_offset, is_local=True)
    pose_quat_global = (R.from_quat(sk.global_rotation.reshape(-1, 4).numpy())
                        * R.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)
    sk = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global),
                                                         root_trans_offset, is_local=False)
    return {"pose_quat_global": sk.global_rotation.numpy(), "pose_quat": sk.local_rotation.numpy(),
            "trans_orig": root_trans, "root_trans_offset": root_trans_offset, "beta": np.zeros(16),
            "gender": "neutral", "pose_aa": pose_aa, "fps": FPS}


# ------------------------------------------------------------------------------------------------ main

EXCLUDE = ("backward", "circle", "turn", "support", "carry", "stair", "slope", "side", "sneak", "stumbl", "jog", "run",
           "egyptian", "steps", "handrail")


def candidates():
    clips = {}
    for f in ("data/kit_train_motion_dict.pkl", "data/kit_test_motion_dict.pkl"):
        clips.update(joblib.load(f))
    out = []
    for k, c in clips.items():
        low = k.lower()
        if "walk" not in low or any(w in low for w in EXCLUDE):
            continue
        tr = np.asarray(c["trans_orig"], float)
        if len(tr) < 3 * FPS:
            continue
        a, b, v = steady_window(tr, tol=0.2)
        if b - a < 1.5 * FPS or straightness(tr, a, b) < 0.95:
            continue
        out.append((k, a, b, v))
    return clips, out


def main(out_dir, speeds=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), seconds=65.0, alternatives=1):
    os.makedirs(out_dir, exist_ok=True)
    clips, cands = candidates()
    print(f"{len(cands)} straight forward walks with a steady stretch >= 1.5 s")
    index, used = {}, set()
    for target in speeds:
        # nearest steady speed (KINESIS plays KIT ~10 % slower than real time: judge by the clip's own frames);
        # a clip already used for another speed is not used again (neighbouring targets would otherwise share clips)
        ranked = sorted((c for c in cands if c[0] not in used), key=lambda c: abs(c[3] - target))[:max(6, 3 * alternatives)]
        scored = []
        for k, a, b, v in ranked:
            c = clips[k]
            pose_aa, tr = np.asarray(c["pose_aa"], float), np.asarray(c["trans_orig"], float)
            (i0, i1), score = best_cycle(pose_aa, a, b)
            scored.append((abs(v - target) + 0.1 * score, k, i0, i1, v, score))
        scored.sort()
        for rank, (_, k, i0, i1, v, score) in enumerate(scored[:alternatives]):
            write_one(out_dir, clips, index, target, rank, k, i0, i1, v, score, seconds)
            used.add(k)
    with open(os.path.join(out_dir, "long_walk_index.json"), "w") as f:
        json.dump(index, f, indent=1)


def write_one(out_dir, clips, index, target, rank, k, i0, i1, v, score, seconds):
    if True:
        c = clips[k]
        pose, tr, reps = build(np.asarray(c["pose_aa"], float), np.asarray(c["trans_orig"], float), i0, i1, seconds)
        motion = to_motion(pose, tr)
        name = f"long_walk_{target:.1f}" + (f"_r{rank}" if rank else "")
        joblib.dump({f"0-{name}": motion}, os.path.join(out_dir, f"{name}.pkl"))
        sp = heading_speed(tr)
        index[name] = {"source": k, "cycle_frames": [int(i0), int(i1)], "cycle_s": (i1 - i0) / FPS,
                       "seam_mismatch_rad": score, "source_steady_speed": v, "repeats": reps,
                       "seconds": len(pose) / FPS, "speed_after_start": float(np.mean(sp[i0:]))}
        print(f"{name}: {k} cycle {i0}-{i1} ({(i1 - i0) / FPS:.2f} s, seam {score:.2f} rad), "
              f"{len(pose) / FPS:.0f} s, {np.mean(sp[i0:]):.2f} m/s")


if __name__ == "__main__":
    sp = tuple(float(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    main(sys.argv[1], sp, float(sys.argv[3]) if len(sys.argv) > 3 else 65.0,
         int(sys.argv[4]) if len(sys.argv) > 4 else 1)
