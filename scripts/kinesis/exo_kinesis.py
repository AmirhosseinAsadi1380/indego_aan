"""Run KINESIS (training or evaluation, its own src/run.py) with the Indego folded into the MyoLegs model.

Every episode reset restores the original model and applies the exo with `kinesis_exo.add_exo` (mass, inertia and
drive friction of the Indego plus springs for its restriction of hip abduction, hip rotation and ankle):

    --exo-mode none     no exo (plain KINESIS)
    --exo-mode fixed    the same exo every episode: --exo-kabd, --exo-krot, --exo-kank (N*m/rad), --exo-mass,
                        --exo-friction (scales), --exo-ank0 (AFO spring neutral, deg dorsiflexion; default 5 as in
                        indego_sim)
    --exo-mode random   a new draw every episode (training): --exo-ranges "kabd=0:600,krot=0:50,kank=0:150,
                        mass=0.8:1.2,friction=0:1,p_none=0.2[,ank0=0:10]" (p_none: share of episodes without the exo;
                        ank0 fixed at --exo-ank0 unless given a range)

Optional deficits (training the person to cope with weakness), a new draw every episode:

    --deficit-ranges "p_drive=0.25,drive=0.4:0.9,drop_at=0:6,ramp=0.5:2,p_fatigue=0.25,fatigue=0:0.5"
        p_drive: share of episodes whose volitional drive drops, at drop_at s, over ramp s, to a level in drive;
        p_fatigue: share of episodes with every muscle's available activation capped at 1 - MF, MF uniform in fatigue.
        The muscle activations of KINESIS's PD law are scaled by the drive and capped (the person "tries harder"
        only through its policy, which sees the resulting tracking errors).

Optional pelvis tracking reward (KINESIS rewards only the positions of pelvis, knees, ankles and toes, so the pelvis
can turn freely: its policy walks with a +-35-40 deg pelvis twist, compensated by hip rotation from one limit to the
other, which a hip-rotation spring such as the Indego's thigh shells resists):

    --pelvis-reward "w_rot=0.1,k_rot=0.2,w_yaw=0.1,k_yaw=5"
        adds w_rot * exp(-k_rot |w_pelvis - w_ref|^2) (pelvis angular velocity vs the reference pelvis', rad/s, world
        frame) and w_yaw * exp(-k_yaw dpsi^2) (heading of the pelvis' forward axis vs the reference pelvis', rad).

All other arguments go to KINESIS's run.py (Hydra overrides). Run from the KINESIS checkout, e.g.

    python <this file> --exo-mode fixed --exo-kabd 300 --config-name config_legs.yaml exp_name=kinesis-moe-imitation \
        epoch=-1 learning.actor_type=moe run=eval_run_legs run.headless=True \
        run.motion_file=data/kit_test_motion_dict.pkl run.initial_pose_file=data/initial_pose/legs/initial_pose_test.pkl \
        env.termination_distance=0.5 run.num_motions=108 no_log=True
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.getcwd())
sys.path.append(HERE)

EXO_DEFAULTS = {"mode": "none", "kabd": 0.0, "krot": 10.0, "kank": 10.0, "mass": 1.0, "friction": 1.0, "ank0": 5.0,
                "ranges": "kabd=0:600,krot=0:50,kank=0:150,mass=0.8:1.2,friction=0:1,p_none=0.2"}
DEFICIT = {"ranges": None, "drive": None, "t_drop": 0.0, "ramp": 1.0, "cap": None, "t0": 0.0}   # this worker's episode
PELVIS = {"w_rot": 0.0, "k_rot": 0.2, "w_yaw": 0.0, "k_yaw": 5.0}
ARRAYS = ("body_mass", "body_ipos", "body_iquat", "body_inertia", "dof_armature", "dof_frictionloss", "dof_damping",
          "jnt_stiffness", "qpos_spring")


def parse_exo_args(argv):
    """Remove --exo-* options from argv (the rest is for Hydra); return the exo settings."""
    opts, rest, i = dict(EXO_DEFAULTS), [argv[0]], 1
    while i < len(argv):
        a = argv[i]
        if a == "--deficit-ranges":
            DEFICIT["ranges"] = {k: tuple(float(x) for x in v.split(":")) if ":" in v else float(v)
                                 for k, v in (kv.split("=") for kv in argv[i + 1].split(","))}
            i += 2
        elif a == "--pelvis-reward":
            PELVIS.update({k: float(v) for k, v in (kv.split("=") for kv in argv[i + 1].split(","))})
            i += 2
        elif a.startswith("--exo-"):
            key = a[len("--exo-"):]
            opts[key] = argv[i + 1]
            i += 2
        else:
            rest.append(a)
            i += 1
    for k in ("kabd", "krot", "kank", "mass", "friction", "ank0"):
        opts[k] = float(opts[k])
    opts["ranges"] = {k: tuple(float(x) for x in v.split(":")) if ":" in v else float(v)
                      for k, v in (kv.split("=") for kv in opts["ranges"].split(","))}
    return opts, rest


def install(opts):
    """Patch KINESIS's imitation environment so every reset applies the exo."""
    import numpy as np
    from kinesis_exo import add_exo
    from src.env.myolegs_im import MyoLegsIm
    orig_reset = MyoLegsIm.reset

    def reset(self, *a, **k):
        m = self.mj_model
        if not hasattr(self, "_exo_base"):
            self._exo_base = {n: getattr(m, n).copy() for n in ARRAYS}
            self._exo_rng = np.random.default_rng((os.getpid() * 7919 + id(self)) % (2 ** 32))
        for n, v in self._exo_base.items():
            getattr(m, n)[:] = v
        mode = opts["mode"]
        if mode == "fixed":
            p = dict(kabd=opts["kabd"], krot=opts["krot"], kank=opts["kank"], mass=opts["mass"],
                     friction=opts["friction"], ank0=opts.get("ank0", 5.0))
        elif mode == "random":
            r, g = opts["ranges"], self._exo_rng
            p = None if g.random() < r.get("p_none", 0.0) else {
                k: g.uniform(*r[k]) for k in ("kabd", "krot", "kank", "mass", "friction")}
            if p is not None:
                p["ank0"] = g.uniform(*r["ank0"]) if "ank0" in r else opts.get("ank0", 5.0)
        else:
            p = None
        if p is not None:
            add_exo(m, self.mj_data, mass_scale=p["mass"], friction_scale=p["friction"], k_hip_abduction=p["kabd"],
                    k_hip_rotation=p["krot"], k_ankle=p["kank"], ankle_neutral_deg=p["ank0"])
        self.exo_params = p
        out = orig_reset(self, *a, **k)
        draw_deficit(self._exo_rng, m.nu, float(self.mj_data.time))
        return out

    MyoLegsIm.reset = reset
    if DEFICIT["ranges"] is not None:
        import src.env.myolegs_env as E
        orig_act = E.target_length_to_activation

        def target_length_to_activation(target_lengths, mj_data, mj_model, *a, **k):
            act = orig_act(target_lengths, mj_data, mj_model, *a, **k)
            d = DEFICIT
            if d["drive"] is not None:
                t = mj_data.time - d["t0"] - d["t_drop"]
                act = act * (1.0 - min(max(t / d["ramp"], 0.0), 1.0) * (1.0 - d["drive"]))
            if d["cap"] is not None:
                act = np.minimum(act, d["cap"])
            return act

        E.target_length_to_activation = target_length_to_activation
    if PELVIS["w_rot"] > 0 or PELVIS["w_yaw"] > 0:
        orig_reward = MyoLegsIm.compute_reward

        def compute_reward(self, action=None):
            reward = orig_reward(self, action)
            r_rot, r_yaw = pelvis_rewards(self)
            self.reward_info["r_pelvis_rot"], self.reward_info["r_pelvis_yaw"] = r_rot, r_yaw
            return reward + PELVIS["w_rot"] * r_rot + PELVIS["w_yaw"] * r_yaw

        MyoLegsIm.compute_reward = compute_reward


def pelvis_rewards(env):
    """(rate, heading) tracking rewards of the pelvis against the reference pelvis at the current time."""
    import numpy as np
    t = env.cur_t * env.dt + env._motion_start_times + env._motion_start_times_offset
    ref = env.get_state_from_motionlib_cache(env._sampled_motion_ids, t, env.global_offset)   # cached: same query
    dw = env.get_body_angular_vel()[0] - ref.root_ang_vel[0]          # root sensor (rigid with the pelvis), world frame
    r_rot = float(np.exp(-PELVIS["k_rot"] * float(dw @ dw)))
    if not hasattr(env, "_pelvis_id"):
        env._pelvis_id = env.mj_model.body("pelvis").id
    x = env.mj_data.xmat[env._pelvis_id].reshape(3, 3)[:, 0]          # MyoLeg pelvis: x forward, y up
    z = quat_wxyz_to_mat(ref.root_rot[0])[:, 2]                       # SMPL pelvis: z forward, y up
    dpsi = np.arctan2(x[1], x[0]) - np.arctan2(z[1], z[0])
    dpsi = (dpsi + np.pi) % (2 * np.pi) - np.pi
    return r_rot, float(np.exp(-PELVIS["k_yaw"] * dpsi ** 2))


def quat_wxyz_to_mat(q):
    """Rotation matrix of a (w, x, y, z) quaternion - the motion library's body rotations are in MuJoCo's order."""
    import numpy as np
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def draw_deficit(rng, n_muscles, t0):
    r = DEFICIT["ranges"]
    DEFICIT.update(drive=None, cap=None, t0=t0)
    if r is None:
        return
    if rng.random() < r.get("p_drive", 0.0):
        DEFICIT.update(drive=rng.uniform(*r["drive"]), t_drop=rng.uniform(*r["drop_at"]), ramp=rng.uniform(*r["ramp"]))
    if rng.random() < r.get("p_fatigue", 0.0):
        DEFICIT["cap"] = 1.0 - rng.uniform(*r["fatigue"], size=n_muscles)


if __name__ == "__main__":
    opts, sys.argv = parse_exo_args(sys.argv)
    print(f"exo: {opts['mode']}" + (f" {opts}" if opts["mode"] != "none" else "")
          + (f"; deficits {DEFICIT['ranges']}" if DEFICIT["ranges"] else "")
          + (f"; pelvis reward {PELVIS}" if PELVIS["w_rot"] > 0 or PELVIS["w_yaw"] > 0 else ""), flush=True)
    install(opts)
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    import runpy
    runpy.run_path(os.path.join(os.getcwd(), "src", "run.py"), run_name="__main__")
