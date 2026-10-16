"""Score existing exo controllers on the three success criteria across the deficit range (the bar for Phase 4).

Controllers: zero torque, transparent (friction + exo-weight compensation), the Indego-like FSM, assist-as-needed
(indego_sim's: impedance to a normative gait template, gain adapting to the error), "paan": assist-as-needed toward
the person's own fresh gait (indego_aan.controllers.PersonalAAN; run scripts/fresh_reference.py first), "teacher": the
PPO teacher in results/teacher (scripts/train_teacher.py; privileged observations) - or, for a folder with only an
env.json, its base controller alone (the tuned teacher, scripts/tune_teacher_aan.py). Deficits: fresh, volitional drive
dropping to 0.6 / 0.4 / 0.3 after 10 s, fatigue accumulating at 30x, pre-fatigued hip and knee extensors (50 %).
Metrics over the last 10 s against the person's fresh, zero-torque baseline.

    python scripts/baselines.py [seeds=0,1,2,3] [controllers=zero,transparent,fsm,aan,paan] [out=results/baselines.json]
        [teacher=results/teacher] [workers=auto]
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFICITS = [("fresh", None), ("drive", 0.6), ("drive", 0.4), ("drive", 0.3), ("fatigue", 30.0), ("prefatigue", 0.5)]
T = 30.0


def controller(name, sim):
    if name == "zero":
        return lambda obs: np.zeros(4)
    if name == "transparent":
        from indego_sim.controllers.transparent import TransparentController
        return TransparentController()
    if name == "aan":
        from indego_sim.controllers.transparent import AssistAsNeeded
        return AssistAsNeeded(dt=sim.ctrl_dt)
    if name == "paan":
        from indego_aan.controllers import PersonalAAN
        from fresh_reference import reference_path
        return PersonalAAN(dict(np.load(reference_path())), dt=sim.ctrl_dt)
    if name == "fsm":
        from indego_sim.controllers.indego_fsm import IndegoFSMController
        return IndegoFSMController.from_model(sim.m, lat_dy=0.0)
    raise ValueError(name)


def attach(kind, level, sim):
    from indego_aan.deficits import Deficits, DriveProcess, Fatigue3CCr
    if kind == "drive":
        return Deficits(sim, drive=DriveProcess(schedule=[(0.0, 1.0), (10.0, 1.0), (15.0, level), (1e9, level)]))
    if kind == "fatigue":
        return Deficits(sim, fatigue=Fatigue3CCr(sim.human.mm.names, rate_scale=level))
    if kind == "prefatigue":
        fat = Fatigue3CCr(sim.human.mm.names)
        fat.reset(np.array([level if g in ("hip_extensors", "knee_extensors") else 0.0 for g in fat.group]))
        return Deficits(sim, fatigue=fat)
    return Deficits(sim)


TEACHER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "teacher")


def run_teacher(kind, level, seed, path):
    import pickle
    import torch
    from stable_baselines3 import PPO
    from indego_aan.env import env_from_config
    from indego_aan.rollout import Recorder
    torch.set_num_threads(1)
    model = norm = None
    if os.path.exists(os.path.join(path, "model.zip")):     # else: the configured base controller alone (tuned teacher)
        model = PPO.load(os.path.join(path, "model.zip"), device="cpu")
        with open(os.path.join(path, "vecnormalize.pkl"), "rb") as f:
            norm = pickle.load(f)
    cfg_path = os.path.join(path, "env.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    env = env_from_config(cfg, deficit={"kind": kind, "level": level}, episode_s=T)
    obs, _ = env.reset(options={"sim_seed": seed})
    env.recorder = Recorder(env.sim, env.defi)
    while True:
        act = np.zeros(4) if model is None else model.predict(norm.normalize_obs(obs), deterministic=True)[0]
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            return env.recorder.arrays()


def run(job):
    ctrl, kind, level, seed = job[:4]
    if ctrl == "teacher":
        return ctrl, kind, level, seed, run_teacher(kind, level, seed, job[4])
    from indego_aan.rollout import run_episode
    from indego_aan.scenario import make_walking_sim
    sim, obs = make_walking_sim(seed=seed)
    defi = attach(kind, level, sim)
    rec = run_episode(sim, controller(ctrl, sim), T, deficits=defi, obs=obs)
    return ctrl, kind, level, seed, rec


if __name__ == "__main__":
    from indego_aan.metrics import GaitBaseline, evaluate
    seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3").split(",")]
    ctrls = (sys.argv[2] if len(sys.argv) > 2 else "zero,transparent,fsm,aan,paan").split(",")
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "results",
                                                                "baselines.json")
    teacher = sys.argv[4] if len(sys.argv) > 4 else TEACHER
    jobs = [(c, k, v, s, teacher) for c in ctrls for k, v in DEFICITS for s in seeds]
    if "zero" not in ctrls:
        jobs += [("zero", "fresh", None, s, teacher) for s in seeds]
    from indego_aan.resources import single_thread_env, worker_count
    single_thread_env()
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else None
    with Pool(worker_count(len(jobs), requested=workers)) as pool:
        res = pool.map(run, jobs)
    base = GaitBaseline.from_records([r for c, k, _, _, r in res if c == "zero" and k == "fresh"], t0=3.0)
    rows = []
    for c, k, v, s, rec in res:
        t_end = rec["t"][-1]
        e = evaluate(rec, base, t0=max(t_end - 10.0, 3.0), t1=t_end) if not rec["fell"] else evaluate(rec, None)
        e.update(controller=c, kind=k, level=v, seed=s)
        rows.append(e)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)
    print(f"fresh baseline (zero torque): speed {base.speed:.2f} m/s, step {base.step_time:.3f} s (CV {base.step_time_cv:.3f})")
    print(f"{'controller':12s} {'deficit':15s} {'falls':>6s} {'speed':>6s} {'stepCV':>7s} {'gaitdev':>8s} {'ankle':>6s} "
          f"{'exoW/s':>7s} {'humW/s':>7s} {'share':>6s} {'tau^2':>7s}")
    for c in ctrls:
        for k, v in DEFICITS:
            g = [r for r in rows if r["controller"] == c and r["kind"] == k and r["level"] == v]
            ok = [r for r in g if not r["fell"]]

            def m(key, f="{:.2f}"):
                vals = [r[key] for r in ok if key in r and not np.isnan(r[key])]
                return f.format(np.mean(vals)) if vals else "-"
            name = k if v is None else f"{k} {v:g}"
            print(f"{c:12s} {name:15s} {sum(r['fell'] for r in g):>3d}/{len(g):<2d} {m('speed'):>6s} "
                  f"{m('step_time_cv', '{:.3f}'):>7s} {m('gait_deviation', '{:.1f}'):>8s} "
                  f"{m('gait_deviation_ankle', '{:.1f}'):>6s} "
                  f"{m('exo_pos_work_per_s', '{:.1f}'):>7s} {m('human_pos_work_per_s', '{:.0f}'):>7s} "
                  f"{m('human_share', '{:.3f}'):>6s} {m('exo_tau_sq', '{:.0f}'):>7s}")
