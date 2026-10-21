"""Serve the pretrained KINESIS legs policy to indego_sim (which runs in another Python / MuJoCo version).

At each 30 Hz decision indego_sim sends the person's state - leg joint angles and velocities (KINESIS and indego_sim
share the MyoLeg leg model: same joints, knee couplings and 80 muscles), the pelvis pose and velocity, and the four
foot contact forces (heel and forefoot, both feet). The server mirrors that state onto KINESIS's own model (the pelvis
moved into the reference clip's frame: heading, position and height aligned at the start), computes KINESIS's
observation against the reference clip at the current time with KINESIS's own code, runs the policy (deterministic)
and returns the 80 target muscle lengths. Physics stays in indego_sim; this process only observes and decides.

Run with the KINESIS environment, from the KINESIS checkout:

    cd <kinesis>; .venv/Scripts/python <this file> <motion_dict.pkl> <initial_poses.pkl> [port=6010] [model_dir]

model_dir: folder with the policy checkpoint (model.pth); default KINESIS's pretrained legs MoE.

Messages (plain Python types; the two numpy versions cannot unpickle each other's arrays):
    {"cmd": "reset", "pelvis_pos": [3], "pelvis_quat": [4] (w, x, y, z)} -> {"ok", "ref_root": [3]}
    {"cmd": "act", "t": s, "q": {joint: angle}, "qd": {joint: velocity}, "pelvis_pos", "pelvis_quat",
     "pelvis_linvel": [3], "pelvis_angvel": [3] (world frame), "touch": [r_foot, r_toes, l_foot, l_toes]}
        -> {"target_length": [80], "ref_error": mean distance of the tracked bodies to the reference (m)}
    {"cmd": "close"}
"""
import os
import re
import sys

sys.path.append(os.getcwd())
os.environ["OMP_NUM_THREADS"] = "1"

from multiprocessing.connection import Listener  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from scipy.spatial.transform import Rotation as sRot  # noqa: E402

from src.agents import agent_dict  # noqa: E402
from src.env.myolegs_env import action_to_target_length  # noqa: E402
from src.learning.policy_moe import PolicyMOE  # noqa: E402

AUTHKEY = b"indego_sim-kinesis"


def indego_name(kinesis_joint):
    """KINESIS joint name -> indego_sim joint name (only the coupled knee joints differ: knee_angle_r_X -> knee_angle_X_r)."""
    m = re.match(r"(knee_angle)_([rl])_(.+)", kinesis_joint)
    return f"{m.group(1)}_{m.group(3)}_{m.group(2)}" if m else kinesis_joint


def yaw_rotation(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Server:
    def __init__(self, motion_file, initial_pose_file, model_dir=None):
        with initialize_config_dir(config_dir=os.path.join(os.getcwd(), "cfg"), version_base=None):
            cfg = compose(config_name="config_legs", overrides=[
                "exp_name=kinesis-moe-imitation", "epoch=-1", "learning.actor_type=moe", "run=eval_run_legs",
                "run.headless=True", f"run.motion_file={motion_file}", f"run.initial_pose_file={initial_pose_file}",
                "env.termination_distance=0.5", "no_log=True", "run.num_motions=1"])
        cfg.output_dir = model_dir or os.path.join(os.getcwd(), "data", "trained_models", "legs", "kinesis-moe-imitation")
        torch.set_default_dtype(torch.float32)
        self.agent = agent_dict[cfg.learning.agent_name](cfg, torch.float32, torch.device("cpu"), training=True,
                                                         checkpoint_epoch=cfg.epoch)
        self.env = self.agent.env
        m = self.env.mj_model
        self.m, self.d = m, self.env.mj_data
        self.joints = [(i, indego_name(m.joint(i).name)) for i in range(m.njnt) if m.jnt_type[i] in (2, 3)]
        root = m.body("root").id
        pelvis = m.body("pelvis").id
        assert m.body_parentid[pelvis] == root, "pelvis is expected to hang rigidly from the root body"
        self.root_jnt = m.body_jntadr[root]
        self.p_rp = m.body_pos[pelvis].copy()                       # pelvis origin in the root frame
        self.R_rp = sRot.from_quat(m.body_quat[pelvis][[1, 2, 3, 0]]).as_matrix()
        self.touch = slice(m.nsensordata - 4, m.nsensordata)        # r_foot, r_toes, l_foot, l_toes (last sensors)

    def reset(self, msg):
        next(iter(self.env.forward_motions()))                      # load the (single) reference clip
        self.env.reset()
        mujoco.mj_kinematics(self.m, self.d)
        pel = self.m.body("pelvis").id
        Rk, pk = self.d.xmat[pel].reshape(3, 3).copy(), self.d.xpos[pel].copy()
        Ri = sRot.from_quat(np.roll(msg["pelvis_quat"], -1)).as_matrix()
        Rrel = Rk @ Ri.T
        self.R = yaw_rotation(np.arctan2(Rrel[1, 0], Rrel[0, 0]))  # indego_sim world -> reference frame (heading)
        self.pi0, self.pk0 = np.asarray(msg["pelvis_pos"]), pk     # positions aligned at the start (incl. height)
        self.env.cur_t = 0
        own = self.env.compute_observations()                      # KINESIS's own start state (diagnostics)
        return {"ok": True, "ref_root": pk.tolist(), "obs_kinesis_start": [float(x) for x in own]}

    def act(self, msg):
        m, d, R = self.m, self.d, self.R
        # pelvis pose and velocity in the reference frame -> root free joint
        p_pel = R @ (np.asarray(msg["pelvis_pos"]) - self.pi0) + self.pk0
        R_pel = R @ sRot.from_quat(np.roll(msg["pelvis_quat"], -1)).as_matrix()
        R_root = R_pel @ self.R_rp.T
        p_root = p_pel - R_root @ self.p_rp
        w = R @ np.asarray(msg["pelvis_angvel"])
        v_root = R @ np.asarray(msg["pelvis_linvel"]) + np.cross(w, p_root - p_pel)
        qa, da = m.jnt_qposadr[self.root_jnt], m.jnt_dofadr[self.root_jnt]
        d.qpos[qa:qa + 3] = p_root
        d.qpos[qa + 3:qa + 7] = np.roll(sRot.from_matrix(R_root).as_quat(), 1)
        d.qvel[da:da + 3] = v_root
        d.qvel[da + 3:da + 6] = R_root.T @ w
        for i, name in self.joints:
            d.qpos[m.jnt_qposadr[i]] = msg["q"][name]
            d.qvel[m.jnt_dofadr[i]] = msg["qd"][name]
        mujoco.mj_forward(m, d)                                     # kinematics + velocity sensors
        d.sensordata[self.touch] = msg["touch"]                     # contact forces measured in indego_sim
        env = self.env
        env.cur_t = int(round(msg["t"] / env.dt))
        obs = env.compute_observations()
        state = self.agent.preprocess_obs(obs)
        with torch.no_grad():
            if isinstance(self.agent.policy_net, PolicyMOE):
                action = self.agent.policy_net.select_action(torch.from_numpy(state).to(torch.float32), True)[0]
            else:
                action = self.agent.policy_net.select_action(torch.from_numpy(state).to(torch.float32), True)
        action = self.agent.preprocess_actions(action[0].numpy())
        # tracking error to the reference at this time (diagnostic)
        ref = env.get_state_from_motionlib_cache(env._sampled_motion_ids, env.cur_t * env.dt
                                                 + env._motion_start_times + env._motion_start_times_offset,
                                                 env.global_offset)
        body = env.get_body_xpos()[env.tracked_bodies_id]
        err = float(np.mean(np.linalg.norm(body - ref.xpos[0, env.smpl_tracked_ids], axis=-1)))
        return {"target_length": [float(x) for x in action_to_target_length(action, m)], "ref_error": err,
                "obs": [float(x) for x in obs] if msg.get("debug") else None}


if __name__ == "__main__":
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 6010
    server = Server(sys.argv[1], sys.argv[2], sys.argv[4] if len(sys.argv) > 4 else None)
    with Listener(("localhost", port), authkey=AUTHKEY) as listener:
        print(f"KINESIS policy server listening on {port}", flush=True)
        with listener.accept() as conn:
            while True:
                msg = conn.recv()
                if msg["cmd"] == "close":
                    break
                conn.send(server.reset(msg) if msg["cmd"] == "reset" else server.act(msg))
