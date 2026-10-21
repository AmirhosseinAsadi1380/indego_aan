"""One evaluation set from single-clip motion files (e.g. the long walking references): KINESIS's evaluation walks
every clip of a motion file in turn and reports the share it follows to the end, so one call covers all references
(its simulation is deterministic: one clip is one sample). Each <name>.pkl needs its start pose next to it as
initial_pose_<name>.pkl (initial_poses.py). Writes <out_motion>, <out_pose> and <out_motion minus .pkl>_names.json
(clip index -> name).

Run with the KINESIS environment, from the KINESIS checkout:

    cd <kinesis>; python <this file> <out_motion.pkl> <out_pose.pkl> <clip1.pkl> [<clip2.pkl> ...]
"""
import json
import os
import sys

import joblib


def main(out_motion, out_pose, files):
    motions, poses, names = {}, {}, []
    for f in files:
        clip = joblib.load(f)
        (key, motion), = clip.items()
        name = os.path.basename(f)[:-len(".pkl")]
        p = joblib.load(os.path.join(os.path.dirname(f), f"initial_pose_{name}.pkl"))
        (pose,) = p.values()                              # {clip index: {start time: pose}} of a one-clip file
        i = len(motions)
        motions[f"{i}-{name}"] = motion
        poses[i] = pose
        names.append(name)
    joblib.dump(motions, out_motion)
    joblib.dump(poses, out_pose)
    with open(out_motion[:-len(".pkl")] + "_names.json", "w") as fh:
        json.dump(names, fh, indent=1)
    print(f"{len(names)} clips -> {out_motion}, {out_pose}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
