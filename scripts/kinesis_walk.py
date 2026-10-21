"""Phase 1: the pretrained KINESIS person walking in indego_sim (healthy subject, 3D model), with or without the Indego.

Starts the KINESIS policy server (scripts/kinesis/policy_server.py) in the KINESIS environment, lets the person settle
standing under indego_sim's own controller, then hands the leg muscles to KINESIS, which follows the reference clip.
Reports how far and how fast the person walked, the mean distance of the tracked bodies to the reference, and falls.

    python scripts/kinesis_walk.py [--exo] [--T 6] [--seed 0] [--kinesis-dir $KINESIS_DIR]
        [--motion data/ref_walk.pkl] [--pose data/initial_pose/legs/initial_pose_ref_walk.pkl]

(--motion / --pose are relative to the KINESIS checkout; see scripts/kinesis/README.md.)
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def start_server(kinesis_dir, motion, pose, port, model_dir=None):
    py = os.path.join(kinesis_dir, ".venv", "Scripts" if os.name == "nt" else "bin", "python")
    server = os.path.join(ROOT, "scripts", "kinesis", "policy_server.py")
    proc = subprocess.Popen([py, server, motion, pose, str(port)] + ([model_dir] if model_dir else []), cwd=kinesis_dir, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:                       # wait until it listens (it prints KINESIS's own logs first)
        if "listening" in line:
            return proc
        if proc.poll() is not None:
            break
    raise RuntimeError("the KINESIS policy server did not start")


def bare_feet(m):
    """Collide with MyoLeg's own foot shapes (capsules + heel ellipsoid, disabled in indego_sim) instead of the shoes."""
    for i in range(m.ngeom):
        name = m.geom(i).name
        if name.startswith(("foot_col", "bofoot_col")):
            m.geom_contype[i], m.geom_conaffinity[i] = 1, 0
        elif name.startswith(("foot_shoe", "bofoot_shoe")):
            m.geom_contype[i], m.geom_conaffinity[i] = 0, 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exo", action="store_true", help="attach the Indego (zero motor torque unless --transparent)")
    ap.add_argument("--transparent", action="store_true",
                    help="indego_sim's transparent mode: friction and exo-weight compensation (needs --exo)")
    ap.add_argument("--integrator", default="euler", choices=["euler", "implicitfast"],
                    help="KINESIS was trained with explicit Euler; under implicit integration it falls even in its own model")
    ap.add_argument("--rigid-trunk", action="store_true",
                    help="stiff lumbar joints, approximating KINESIS's rigid torso")
    ap.add_argument("--bare-feet", action="store_true",
                    help="MyoLeg's own foot collision shapes (as KINESIS was trained with) instead of indego_sim's shoes")
    ap.add_argument("--exo-set", default="", help="ExoParams overrides for ablations, e.g. "
                    "'hip_rot_stiffness=50,afo_type=free'")
    ap.add_argument("--record", default=None,
                    help="save the walk as a trial (.npz) for scripts/waterloo_indego.py --trial (needs --exo)")
    ap.add_argument("--T", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=6010)
    ap.add_argument("--kinesis-dir", default=os.environ.get("KINESIS_DIR", os.path.expanduser("~/kinesis")))
    ap.add_argument("--motion", default="data/ref_walk.pkl")
    ap.add_argument("--pose", default="data/initial_pose/legs/initial_pose_ref_walk.pkl")
    ap.add_argument("--model-dir", default=None, help="policy checkpoint folder (default: KINESIS's pretrained legs MoE)")
    a = ap.parse_args()

    from indego_aan.resources import single_thread_env
    single_thread_env()
    from indego_sim.config import SimConfig, SupportParams
    from indego_sim.human.sci import get_preset
    from indego_sim.sim import HumanExoSim
    from indego_aan.kinesis_human import KinesisHuman

    server = start_server(a.kinesis_dir, a.motion, a.pose, a.port, a.model_dir)
    try:
        from indego_sim.config import ExoParams
        exo_p = ExoParams()
        for item in filter(None, a.exo_set.split(",")):
            k, v = item.split("=")
            cur = getattr(exo_p, k)                                   # unknown names fail here
            setattr(exo_p, k, type(cur)(v) if not isinstance(cur, str) else v)
        sim = HumanExoSim(exo=exo_p, subject=get_preset("healthy"), support=SupportParams(kind="none"),
                          sim=SimConfig(planar=False), with_exo=a.exo, seed=a.seed, log_muscles=False, log_rate_hz=20)
        import mujoco
        sim.m.opt.integrator = (mujoco.mjtIntegrator.mjINT_EULER if a.integrator == "euler"
                                else mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        if a.bare_feet:
            bare_feet(sim.m)
        if a.rigid_trunk:
            for j in ("lumbar_flexion", "lumbar_bending", "lumbar_rotation"):
                jid = sim.m.joint(j).id
                sim.m.jnt_stiffness[jid] = 3000.0                 # N*m/rad about the keyframe posture
                sim.m.qpos_spring[sim.m.jnt_qposadr[jid]] = sim.m.key("stand").qpos[sim.m.jnt_qposadr[jid]]
                sim.m.dof_damping[sim.m.jnt_dofadr[jid]] = 100.0
        # without the exo nothing holds the person during indego_sim's settle: start from the standing keyframe at rest
        sim.reset(settle=1.0 if a.exo else 0.0)
        person = KinesisHuman(sim, port=a.port)
        person.start()
        d, pel = sim.d, sim.m.body("pelvis").id
        x0, t0, wall = float(d.xpos[pel][0]), d.time, time.time()
        xs = []
        ctrl = None
        if a.exo and a.transparent:
            from indego_sim.controllers.transparent import TransparentController
            ctrl = TransparentController()
        obs = sim.observe()
        rec = None
        if a.record and a.exo:
            from indego_aan.waterloo import TrialRecorder
            rec = TrialRecorder(sim)
        hips = [sim.m.jnt_qposadr[sim.m.joint(j).id] for j in ("hip_rotation_r", "hip_rotation_l", "hip_adduction_r",
                                                                  "hip_adduction_l")]
        pose = []                                                 # pelvis orientation / position, hip rotation, adduction
        for _ in range(int(round(a.T / sim.ctrl_dt))):
            tau = (ctrl(obs) if ctrl else np.zeros(4)) if a.exo else None
            obs = sim.step(tau, record=False)
            if rec is not None:
                rec(obs)
            xs.append((d.time - t0, float(d.xpos[pel][0]), float(d.xpos[pel][1])))
            pose.append(np.concatenate([d.xmat[pel], d.xpos[pel], np.degrees(d.qpos[hips])]))
            if sim.fell:
                break
        xs = np.array(xs)
        err = np.array(person.ref_error)
        mid = (xs[:, 0] > xs[-1, 0] * 0.25) & (xs[:, 0] < xs[-1, 0] * 0.75)
        speed = np.polyfit(xs[mid, 0], xs[mid, 1], 1)[0] if mid.sum() > 10 else np.nan
        print(f"{('with the Indego (transparent)' if a.transparent else 'with the Indego (zero torque)') if a.exo else 'without the Indego'}{', bare feet' if a.bare_feet else ', shoes'}, {a.integrator}{', rigid trunk' if a.rigid_trunk else ''}: {'FELL at %.1f s' % xs[-1, 0] if sim.fell else 'walked %.1f s' % xs[-1, 0]}, "
              f"forward {xs[-1, 1] - x0:.2f} m, sideways {xs[-1, 2] - xs[0, 2]:+.2f} m, mid-clip speed {speed:.2f} m/s; "
              f"distance to the reference: mean {err[:, 1].mean() * 100:.1f} cm, max {err[:, 1].max() * 100:.1f} cm "
              f"[{time.time() - wall:.0f} s wall]")
        if xs[-1, 0] > 6.0:
            from indego_aan.waterloo import pelvis_ranges
            pose = np.array(pose)
            pr = pelvis_ranges(xs[:, 0], pose[:, :9].reshape(-1, 3, 3), pose[:, 9:12])
            w = xs[:, 0] > 3.0
            hr = [float(np.ptp(np.percentile(pose[w, 12 + i], [2, 98]))) for i in range(4)]
            print(f"pelvis yaw / pitch / roll range {pr['pelvis_yaw_range']:.1f} / {pr['pelvis_pitch_range']:.1f} / "
                  f"{pr['pelvis_roll_range']:.1f} deg (people ~10-15 / 3-5 / 5-8); hip rotation range R/L "
                  f"{hr[0]:.1f}/{hr[1]:.1f}, adduction {hr[2]:.1f}/{hr[3]:.1f} deg")
        if rec is not None:
            np.savez_compressed(a.record, **rec.arrays(person="kinesis", seed=a.seed, motion=a.motion,
                                                       model=str(a.model_dir), exo_set=a.exo_set))
            print("trial ->", a.record)
        person.close()
    finally:
        server.terminate()
