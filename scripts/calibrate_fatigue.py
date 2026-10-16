"""Phase 2 calibration: which pre-fatigue state reproduces DUO-GAIT's fresh -> fatigued gait changes?

DUO-GAIT's fatigue comes from sit-to-stand to exhaustion, which loads the hip and knee extensors (and, less, the
ankle plantarflexors). Here the walk starts with those muscle groups partly fatigued (3CC-r MF, recovering and
re-fatiguing at the published rates while walking, zero-torque exo); speed, step time, stride length and step-time
variability over the walk are compared, seed by seed, with fresh walks of the same person, and the relative changes
with DUO-GAIT's (results/duo_gait_targets.json, scripts/duo_gait.py).

    python scripts/calibrate_fatigue.py [seeds=0,1,2,3] [T=60] -> results/phase2_calibration.json
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
GROUPS = {"hip_knee_ext": ("hip_extensors", "knee_extensors"),
          "hip_knee_ext_plantar": ("hip_extensors", "knee_extensors", "ankle_plantarflexors")}
T0 = 5.0                                  # s: skip gait initiation


def run(job):
    groups, level, seed, T = job
    from indego_aan.deficits import Deficits, Fatigue3CCr
    from indego_aan.rollout import run_episode
    from indego_aan.scenario import make_walking_sim
    sim, obs = make_walking_sim(seed=seed)
    fat = Fatigue3CCr(sim.human.mm.names)
    if groups is not None:
        fat.reset(np.array([level if g in GROUPS[groups] else 0.0 for g in fat.group]))
    rec = run_episode(sim, lambda o: np.zeros(4), T, deficits=Deficits(sim, fatigue=fat), obs=obs)
    return groups, level, seed, rec


def gait_stats(rec):
    from indego_aan.metrics import evaluate
    e = evaluate(rec, None, t0=T0, t1=rec["t"][-1])
    return {"speed": e["speed"], "step_time": e["step_time"], "stride_length": 2.0 * e["speed"] * e["step_time"],
            "step_time_cv": e["step_time_cv"]}


if __name__ == "__main__":
    seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3").split(",")]
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    jobs = [(None, 0.0, s, T) for s in seeds] + [(g, v, s, T) for g in GROUPS for v in LEVELS for s in seeds]
    from indego_aan.resources import single_thread_env, worker_count
    single_thread_env()
    with Pool(worker_count(len(jobs))) as pool:
        res = pool.map(run, jobs)
    fresh = {s: gait_stats(r) for g, _, s, r in res if g is None}
    tgt = json.load(open(os.path.join(ROOT, "results", "duo_gait_targets.json")))
    target = {"speed": tgt["speed_avg"]["rel_change_mean"], "step_time": tgt["stride_times_avg"]["rel_change_mean"],
              "stride_length": tgt["stride_lengths_avg"]["rel_change_mean"],
              "step_time_cv": tgt["stride_times_CV"]["rel_change_mean"]}
    rows = []
    for g, v, s, rec in res:
        if g is None:
            continue
        row = {"groups": g, "level": v, "seed": s, "fell": bool(rec["fell"])}
        if not rec["fell"]:
            st = gait_stats(rec)
            row.update({f"rel_{k}": st[k] / fresh[s][k] - 1.0 for k in st})
        rows.append(row)
    json.dump({"target": target, "fresh": fresh, "rows": rows}, open(os.path.join(ROOT, "results",
                                                                             "phase2_calibration.json"), "w"), indent=1)
    keys = list(target)
    print("relative change vs fresh (mean over seeds that did not fall)")
    print(f"{'groups':22s} {'MF0':>4s} {'falls':>6s} " + " ".join(f"{k:>14s}" for k in keys))
    print(f"{'DUO-GAIT':22s} {'':>4s} {'':>6s} " + " ".join(f"{100 * target[k]:+13.1f}%" for k in keys))
    for g in GROUPS:
        for v in LEVELS:
            grp = [r for r in rows if r["groups"] == g and r["level"] == v]
            ok = [r for r in grp if not r["fell"]]
            vals = [np.nanmean([r[f"rel_{k}"] for r in ok]) if ok else np.nan for k in keys]
            print(f"{g:22s} {v:4.1f} {sum(r['fell'] for r in grp):>3d}/{len(grp):<2d} "
                  + " ".join(f"{100 * x:+13.1f}%" for x in vals))
