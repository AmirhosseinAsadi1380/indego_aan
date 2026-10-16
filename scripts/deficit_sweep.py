"""Phase 2: how the unassisted person's gait changes with low volitional drive and with fatigue (zero-torque exo).

Each run walks fresh for 10 s, then the deficit sets in (drive ramps down over 5 s to its level / fatigue
accumulates from the start / "prefatigue": the walk starts with the hip and knee extensors already fatigued, as after
DUO-GAIT's sit-to-stand protocol), and the last 10 s are compared with the person's fresh baseline (separate fresh
runs of the same person).

    python scripts/deficit_sweep.py [seeds=0,1,2,3] [mode=capacity|command] [human=robust|tuned]
        [out=results/phase2_unassisted_<human>_<mode>.json]
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DRIVES = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3)
FATIGUE_RATES = (10.0, 30.0, 100.0)
PREFATIGUE = (0.3, 0.5)                       # fatigued fraction of the hip and knee extensors at the start
T_DRIVE, T_FATIGUE = 30.0, 60.0


def run(job):
    kind, level, seed, mode, human = job
    from indego_aan.deficits import Deficits, DriveProcess, Fatigue3CCr
    from indego_aan.rollout import run_episode
    from indego_aan.scenario import make_walking_sim
    sim, obs = make_walking_sim(seed=seed, human=human)
    if kind == "drive":
        defi = Deficits(sim, drive=DriveProcess(schedule=[(0.0, 1.0), (10.0, 1.0), (15.0, level), (1e9, level)]),
                        mode=mode)
        T = T_DRIVE
    elif kind == "fatigue":
        defi = Deficits(sim, fatigue=Fatigue3CCr(sim.human.mm.names, rate_scale=level), mode=mode)
        T = T_FATIGUE
    elif kind == "prefatigue":
        fat = Fatigue3CCr(sim.human.mm.names)
        fat.reset(np.array([level if g in ("hip_extensors", "knee_extensors") else 0.0 for g in fat.group]))
        defi, T = Deficits(sim, fatigue=fat, mode=mode), T_DRIVE
    else:
        defi, T = Deficits(sim), T_DRIVE
    rec = run_episode(sim, lambda o: np.zeros(4), T, deficits=defi, obs=obs)
    return kind, level, seed, mode, rec


if __name__ == "__main__":
    from indego_aan.metrics import GaitBaseline, evaluate
    seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3").split(",")]
    mode = sys.argv[2] if len(sys.argv) > 2 else "capacity"
    human = sys.argv[3] if len(sys.argv) > 3 else "robust"
    out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(__file__), "..", "results",
                                                                f"phase2_unassisted_{human}_{mode}.json")
    jobs = [("fresh", 1.0, s, mode, human) for s in seeds] + \
        [("drive", v, s, mode, human) for v in DRIVES for s in seeds] + \
        [("fatigue", v, s, mode, human) for v in FATIGUE_RATES for s in seeds] + \
        [("prefatigue", v, s, mode, human) for v in PREFATIGUE for s in seeds]
    from indego_aan.resources import single_thread_env, worker_count
    single_thread_env()
    with Pool(worker_count(len(jobs))) as pool:
        res = pool.map(run, jobs)
    base = GaitBaseline.from_records([r for k, _, _, _, r in res if k == "fresh"], t0=3.0)
    print(f"fresh baseline: {base.n_cycles} cycles, speed {base.speed:.2f} m/s, step {base.step_time:.3f} s "
          f"(CV {base.step_time_cv:.3f})")
    rows = []
    for kind, level, seed, _, rec in res:
        T = rec["t"][-1]
        e = evaluate(rec, base, t0=max(T - 10.0, 3.0), t1=T) if not rec["fell"] else evaluate(rec, None)
        e.update(kind=kind, level=level, seed=seed)
        rows.append(e)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)
    print(f"{'condition':18s} {'falls':>6s} {'speed':>7s} {'step':>7s} {'stepCV':>7s} {'gaitdev':>8s} {'fatigue':>8s}")
    for kind, levels in (("fresh", (1.0,)), ("drive", DRIVES), ("fatigue", FATIGUE_RATES), ("prefatigue", PREFATIGUE)):
        for v in levels:
            g = [r for r in rows if r["kind"] == kind and r["level"] == v]
            ok = [r for r in g if not r["fell"]]

            def m(key):
                vals = [r[key] for r in ok if not np.isnan(r.get(key, np.nan))]
                return f"{np.mean(vals):.3f}" if vals else "-"
            name = f"{kind} {v:g}"
            print(f"{name:18s} {sum(r['fell'] for r in g):>3d}/{len(g):<2d} {m('speed'):>7s} {m('step_time'):>7s} "
                  f"{m('step_time_cv'):>7s} {m('gait_deviation'):>8s} {m('fatigue'):>8s}")
