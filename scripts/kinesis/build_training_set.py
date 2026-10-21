# Uses KINESIS (github.com/amathislab/Kinesis), Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI,
# EPFL, BSD-3-Clause: the start-pose computation follows its src/utils/initial_pose.py. Redistribution and use in
# source and binary forms, with or without modification, are permitted provided that the copyright notices, this list
# of conditions and the disclaimer are retained. THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""A KINESIS training set with sustained walking: KIT's training clips plus the long walking references
(long_reference.py), each long reference repeated so that it is drawn often enough (KINESIS samples clips
uniformly), and the start poses every 0.2 s that random-start training needs (KIT's own for the KIT clips, computed
here for the long references).

Run with the KINESIS environment, from the KINESIS checkout:

    cd <kinesis>; .venv/Scripts/python <this file> <long_walks_dir> <out_motion.pkl> <out_initial_pose.pkl> [repeats=8]
"""
import glob
import os
import sys

sys.path.append(os.getcwd())

import joblib  # noqa: E402
import numpy as np  # noqa: E402


def start_poses(motion_file, config="config_legs"):
    """{start time (0.2 s steps): pose} for the single clip in motion_file, as KINESIS's initial_pose.py."""
    from hydra import compose, initialize_config_dir
    from src.env.myolegs_im import MyoLegsIm
    with initialize_config_dir(config_dir=os.path.join(os.getcwd(), "cfg"), version_base=None):
        cfg = compose(config_name=config, overrides=["run=eval_run_legs", f"run.motion_file={motion_file}",
                                                     "run.initial_pose_file=none.pkl", "run.headless=True"])
    env = MyoLegsIm(cfg)
    env.initial_pos_data = {}
    env.motion_lib.load_motions(env.motion_lib_cfg, shape_params=env.gender_betas, random_sample=False, start_idx=0)
    length = env.motion_lib._motion_lengths[0]
    poses = {}
    for t in np.arange(0, length - 0.5, 0.2):
        env.initial_pose = None
        env.reset(options={"start_time": t})
        poses[float(np.round(t, 1))] = env.initial_pose
    return poses


def main(long_dir, out_motion, out_pose, repeats=8):
    motions = joblib.load("data/kit_train_motion_dict.pkl")
    poses_kit = joblib.load("data/initial_pose/legs/initial_pose_train.pkl")
    out_m = dict(motions)
    out_p = {i: poses_kit[i] for i in range(len(motions)) if i in poses_kit}
    files = sorted(glob.glob(os.path.join(long_dir, "long_walk_*.pkl")))
    for f in files:
        clip = joblib.load(f)
        (name, motion), = clip.items()
        print(f"start poses for {os.path.basename(f)} ...", flush=True)
        poses = start_poses(f)
        for r in range(repeats):
            idx = len(out_m)
            out_m[f"{name}_rep{r}"] = motion
            out_p[idx] = poses
    joblib.dump(out_m, out_motion)
    joblib.dump(out_p, out_pose)
    print(f"{len(motions)} KIT clips + {len(files)} long references x {repeats} = {len(out_m)} clips "
          f"-> {out_motion}, {out_pose}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 8)
