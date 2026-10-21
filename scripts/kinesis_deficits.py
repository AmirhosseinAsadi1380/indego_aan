"""The learned human under deficits: the KINESIS person (exo fine-tune) walking in the Indego on a long reference,
with the same deficits as scripts/baselines.py (fresh; drive dropping to 0.6 / 0.4 after 10-15 s; fatigue at 30x;
pre-fatigued hip and knee extensors), with zero torque and with the personal AAN. The deficits act on the person's
muscle commands (indego_aan.deficits.CommandDeficits). The AAN's reference is this person's own fresh zero-torque
walk (stage 1); its gains are logged next to the true capacity for the deficit estimate (as scripts/deficit_estimate.py).

Assumption: the exo's frontal-plane hip springs are relaxed (abduction and rotation stiffness 0, +-6 deg limit
kept), the setting in which the exo fine-tuned policy walks for a minute (with the springs on it falls; see the
README).

    python scripts/kinesis_deficits.py --model-dir <exo_moe folder> [--speed 0.8] [--seeds 0,1,2,3] [--T 30]
        [--workers 3] [--out results/kinesis_deficits.json]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFICITS = [("fresh", None), ("drive", 0.8), ("drive", 0.6), ("drive", 0.4), ("fatigue", 10.0), ("fatigue", 30.0),
            ("prefatigue", 0.2), ("prefatigue", 0.5)]      # milder levels too: the learned human tolerates less
EXO_SET = {"hip_abd_stiffness": 0.0, "hip_rot_stiffness": 0.0}
EVERY = 10                      # control steps (200 Hz) per logged sample -> 20 Hz
GROUPS = ("hip_extensors", "hip_flexors", "knee_extensors", "knee_flexors")


def make_deficits(kind, level, sim):
    from indego_aan.deficits import CommandDeficits, DriveProcess, Fatigue3CCr
    if kind == "drive":
        return CommandDeficits(sim, drive=DriveProcess(schedule=[(0.0, 1.0), (10.0, 1.0), (15.0, level),
                                                                 (1e9, level)]))
    if kind == "fatigue":
        return CommandDeficits(sim, fatigue=Fatigue3CCr(sim.human.mm.names, rate_scale=level))
    if kind == "prefatigue":
        fat = Fatigue3CCr(sim.human.mm.names)
        fat.reset(np.array([level if g in ("hip_extensors", "knee_extensors") else 0.0 for g in fat.group]))
        return CommandDeficits(sim, fatigue=fat)
    return CommandDeficits(sim)


def capacity(defi):
    fat = defi.fatigue.group_mean() if defi.fatigue is not None else {}
    return defi.drive_now * (1.0 - float(np.mean([fat.get(g, 0.0) for g in GROUPS])))


def run(job):
    ctrl, kind, level, seed, port, a = job
    from indego_aan.resources import single_thread_env
    single_thread_env()
    import mujoco
    from indego_sim.config import ExoParams, SimConfig, SupportParams
    from indego_sim.human.sci import get_preset
    from indego_sim.sim import HumanExoSim
    from indego_aan.kinesis_human import KinesisHuman
    from indego_aan.rollout import Recorder
    from kinesis_walk import start_server
    motion = f"data/long_walks/long_walk_{a['speed']}.pkl"
    pose = f"data/long_walks/initial_pose_long_walk_{a['speed']}.pkl"
    server = start_server(a["kinesis_dir"], motion, pose, port, a["model_dir"])
    try:
        exo_p = ExoParams()
        for k, v in EXO_SET.items():
            setattr(exo_p, k, v)
        sim = HumanExoSim(exo=exo_p, subject=get_preset("healthy"), support=SupportParams(kind="none"),
                          sim=SimConfig(planar=False), with_exo=True, seed=seed, log_muscles=False, log_rate_hz=1)
        sim.m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        for j in ("lumbar_flexion", "lumbar_bending", "lumbar_rotation"):           # rigid trunk, as KINESIS's
            jid = sim.m.joint(j).id
            sim.m.jnt_stiffness[jid] = 3000.0
            sim.m.qpos_spring[sim.m.jnt_qposadr[jid]] = sim.m.key("stand").qpos[sim.m.jnt_qposadr[jid]]
            sim.m.dof_damping[sim.m.jnt_dofadr[jid]] = 100.0
        sim.reset(settle=1.0)
        defi = make_deficits(kind, level, sim)
        person = KinesisHuman(sim, port=port, deficits=defi)
        person.start()
        if ctrl == "paan":
            from indego_aan.controllers import PersonalAAN
            controller = PersonalAAN(dict(np.load(a["reference"])), dt=sim.ctrl_dt)
        else:
            controller = lambda o: np.zeros(4)                                       # noqa: E731
        rec = Recorder(sim, defi)
        obs, rows, tau_ema, n = sim.observe(), [], 0.0, 0
        last_wrap, stride, ph_prev = None, np.nan, None
        for _ in range(int(round(a["T"] / sim.ctrl_dt))):
            tau = controller(obs)
            obs = sim.step(tau, record=False)
            rec()
            n += 1
            if ctrl == "paan":
                tau_ema += 0.01 * (float(np.mean(np.abs(tau))) - tau_ema)
                ph = controller.phase
                if ph is not None and ph_prev is not None and ph < ph_prev - 0.5:
                    t_now = sim.d.time - sim.t0
                    if last_wrap is not None:
                        stride = t_now - last_wrap
                    last_wrap = t_now
                ph_prev = ph
                if n % EVERY == 0:
                    rows.append(dict(t=sim.d.time - sim.t0, cap=capacity(defi), drive=defi.drive_now,
                                     k_hip=float(np.mean(controller.k[[0, 2]])),
                                     k_knee=float(np.mean(controller.k[[1, 3]])), tau_abs=tau_ema, stride_time=stride))
            if sim.fell:
                break
        person.close()
        return ctrl, kind, level, seed, rec.arrays(), rows
    finally:
        server.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--speed", default="0.8")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--T", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--kinesis-dir", default=os.environ.get("KINESIS_DIR", os.path.expanduser("~/kinesis")))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "kinesis_deficits.json"))
    a = ap.parse_args()
    from indego_aan.metrics import GaitBaseline, evaluate
    seeds = [int(s) for s in a.seeds.split(",")]
    ref_path = os.path.join(ROOT, "results", f"reference_kinesis_{a.speed}.npz")
    common = dict(speed=a.speed, kinesis_dir=a.kinesis_dir, model_dir=a.model_dir, T=a.T, reference=ref_path)
    port = iter(range(a.port, a.port + 1000))
    # stage 1: the fresh, zero-torque walk -> the person's own reference (and the metrics' baseline)
    with Pool(min(a.workers, len(seeds))) as pool:
        fresh = pool.map(run, [("zero", "fresh", None, s, next(port), common) for s in seeds])
    recs = [r for *_, r, _ in fresh]
    if any(r["fell"] for r in recs):
        print("WARNING: the fresh person fell in", [s for (_, _, _, s, r, _) in fresh if r["fell"]])
    ok = [r for r in recs if not r["fell"]]
    b = GaitBaseline.from_records(ok, t0=3.0, key="q_exo")
    h = GaitBaseline.from_records(ok, t0=3.0, key="q")
    np.savez(ref_path, mean=b.mean, sd=b.sd, step_time=b.step_time, speed=b.speed, n_cycles=b.n_cycles,
             mean_h=h.mean, sd_h=h.sd, chan_h=np.array(h.chan))
    print(f"fresh reference: {b.n_cycles} cycles, step {b.step_time:.3f} s, speed {b.speed:.2f} m/s -> {ref_path}")
    # stage 2: every deficit with zero torque and with the personal AAN
    jobs = [(c, k, v, s, next(port), common) for c in ("zero", "paan") for k, v in DEFICITS for s in seeds
            if not (c == "zero" and k == "fresh")]
    with Pool(a.workers) as pool:
        res = fresh + pool.map(run, jobs)
    base = GaitBaseline.from_records(ok, t0=3.0)
    rows, feats = [], []
    for c, k, v, s, rec, f in res:
        t_end = rec["t"][-1]
        e = evaluate(rec, base, t0=max(t_end - 10.0, 3.0), t1=t_end) if not rec["fell"] else evaluate(rec, None)
        e.update(controller=c, kind=k, level=v, seed=s, t_end=float(t_end))
        rows.append(e)
        if f:
            feats.append(dict(kind=k, level=v, seed=s, fell=bool(rec["fell"]), rows=f))
    json.dump({"assumption": "exo frontal-plane hip springs relaxed", "speed": a.speed, "model": a.model_dir,
               "rows": rows, "estimator_features": feats}, open(a.out, "w"), indent=1, default=float)
    print(f"{'controller':10s} {'deficit':14s} {'falls':>6s} {'speed':>6s} {'stepCV':>7s} {'gaitdev':>8s} "
          f"{'exoW/s':>7s} {'humW/s':>7s} {'share':>6s}")
    for c in ("zero", "paan"):
        for k, v in DEFICITS:
            g = [r for r in rows if r["controller"] == c and r["kind"] == k and r["level"] == v]
            okr = [r for r in g if not r["fell"]]

            def m(key, f="{:.2f}"):
                vals = [r[key] for r in okr if key in r and not np.isnan(r[key])]
                return f.format(np.mean(vals)) if vals else "-"
            name = k if v is None else f"{k} {v:g}"
            print(f"{c:10s} {name:14s} {sum(r['fell'] for r in g):>3d}/{len(g):<2d} {m('speed'):>6s} "
                  f"{m('step_time_cv', '{:.3f}'):>7s} {m('gait_deviation', '{:.1f}'):>8s} "
                  f"{m('exo_pos_work_per_s', '{:.1f}'):>7s} {m('human_pos_work_per_s', '{:.0f}'):>7s} "
                  f"{m('human_share', '{:.3f}'):>6s}")


if __name__ == "__main__":
    main()
