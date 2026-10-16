# Adapted from KINESIS (src/utils/convert_kit.py), Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL,
# BSD-3-Clause; KINESIS adapts PHC (Copyright (c) 2023 Carnegie Mellon University; Copyright (c) 2018-2023, NVIDIA
# Corporation), BSD-3-Clause. Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the copyright notices, this list of conditions and the disclaimer are retained (see
# github.com/amathislab/Kinesis/blob/main/LICENSE.md). THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""Convert AMASS clips (SMPL+H G .npz) to a KINESIS motion dictionary - the processing of KINESIS's
src/utils/convert_kit.py (30 fps, neutral body, upright start), for any AMASS subset and with Windows paths.

Run with the KINESIS environment, from the KINESIS checkout (it imports KINESIS's poselib and SMPL robot):

    cd <kinesis>; .venv/Scripts/python <this file> <amass_subset_dir> <out.pkl> [name_filter=walk] [exclude=backward,...]

Keys are "0-<SUBSET>_<folder>_<clip>" as in KINESIS. The AMASS / SMPL data and the output stay local (their
licences do not allow redistribution).
"""
import glob
import os
import sys

sys.path.append(os.getcwd())

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation as sRot  # noqa: E402

from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree  # noqa: E402
from src.utils.smpl_skeleton.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES  # noqa: E402
from src.utils.smpl_skeleton.smpl_local_robot import SMPL_Robot as LocalRobot  # noqa: E402

ROBOT_CFG = {"mesh": False, "rel_joint_lm": True, "upright_start": True, "remove_toe": False, "real_weight": True,
             "real_weight_porpotion_capsules": True, "real_weight_porpotion_boxes": True, "replace_feet": True,
             "masterfoot": False, "big_ankle": True, "freeze_hand": False, "box_body": False, "master_range": 50,
             "body_params": {}, "joint_params": {}, "geom_params": {}, "actuator_params": {}, "model": "smpl"}


def convert(src, out, name_filter="walk", exclude=("backward", "back")):
    subset = os.path.basename(os.path.normpath(src))
    robot = LocalRobot(ROBOT_CFG)
    beta = np.zeros(16)
    robot.load_from_skeleton(betas=torch.from_numpy(beta[None]), gender=[0], objs_info=None)
    robot.write_xml("data/xml/smpl_humanoid.xml")
    tree = SkeletonTree.from_mjcf("data/xml/smpl_humanoid.xml")
    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
    motions = {}
    for path in sorted(glob.glob(os.path.join(src, "**", "*.npz"), recursive=True)):
        folder, clip = os.path.normpath(path).split(os.sep)[-2:]
        name = f"{folder}_{clip}".lower()
        if name_filter and name_filter not in name or any(e in name for e in exclude):
            continue
        d = dict(np.load(path, allow_pickle=True))
        if "mocap_framerate" not in d:
            continue
        skip = int(d["mocap_framerate"] / 30)
        trans = d["trans"][::skip]
        pose_aa = np.concatenate([d["poses"][::skip, :66], np.zeros((trans.shape[0], 6))], axis=-1)
        n = pose_aa.shape[0]
        if n < 10:
            continue
        quat = sRot.from_rotvec(pose_aa.reshape(n, 24, 3)[:, smpl_2_mujoco].reshape(-1, 3)).as_quat().reshape(n, 24, 4)
        offset = torch.from_numpy(trans) + tree.local_translation[0]
        st = SkeletonState.from_rotation_and_root_translation(tree, torch.from_numpy(quat), offset, is_local=True)
        qg = (sRot.from_quat(st.global_rotation.reshape(-1, 4).numpy())
              * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(n, -1, 4)
        st = SkeletonState.from_rotation_and_root_translation(tree, torch.from_numpy(qg), offset, is_local=False)
        key = f"0-{subset}_{folder}_{clip}".replace(".npz", "").replace(" ", "_")
        motions[key] = {"pose_quat_global": st.global_rotation.numpy(), "pose_quat": st.local_rotation.numpy(),
                        "trans_orig": trans, "root_trans_offset": offset, "beta": beta, "gender": "neutral",
                        "pose_aa": pose_aa, "fps": 30}
    joblib.dump(motions, out)
    print(f"{len(motions)} clips -> {out}")
    return motions


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "walk",
            tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else ("backward", "back"))
