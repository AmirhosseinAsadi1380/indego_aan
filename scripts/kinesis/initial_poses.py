# Adapted from KINESIS (src/utils/initial_pose.py), Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL,
# BSD-3-Clause; KINESIS adapts PHC (Copyright (c) 2023 Carnegie Mellon University; Copyright (c) 2018-2023, NVIDIA
# Corporation), BSD-3-Clause. Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the copyright notices, this list of conditions and the disclaimer are retained (see
# github.com/amathislab/Kinesis/blob/main/LICENSE.md). THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""Initial poses (muscle model posed to match the first frame of each clip) for a KINESIS motion dictionary - as
KINESIS's src/utils/initial_pose.py, but only for the clip start (what its evaluation uses). Without this file the
evaluation reuses the first clip's pose and position for every later clip.

Run with the KINESIS environment, from the KINESIS checkout:

    cd <kinesis>; .venv/Scripts/python <this file> <motion_dict.pkl> <out.pkl> [config=config_legs]
"""
import os
import sys

sys.path.append(os.getcwd())

import joblib  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from src.env.myolegs_im import MyoLegsIm  # noqa: E402


def main(motion_file, out, config="config_legs"):
    n = len(joblib.load(motion_file))
    with initialize_config_dir(config_dir=os.path.join(os.getcwd(), "cfg"), version_base=None):
        cfg = compose(config_name=config, overrides=["run=eval_run_legs", f"run.motion_file={motion_file}",
                                                     "run.initial_pose_file=none.pkl", "run.headless=True"])
    env = MyoLegsIm(cfg)
    env.initial_pos_data = {}
    poses = {}
    for i in range(n):
        env.motion_lib.load_motions(env.motion_lib_cfg, shape_params=env.gender_betas, random_sample=False,
                                    start_idx=i)
        env.initial_pose = None
        env.reset(options={"start_time": 0.0})
        poses[env.motion_lib._curr_motion_ids[0]] = {0.0: env.initial_pose}
        print(f"{i + 1}/{n}: distance to reference {env.final_distance_to_ref:.3f}", flush=True)
    joblib.dump(poses, out)
    print(f"{len(poses)} initial poses -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "config_legs")
