"""Tune the person's own walking (indego_sim HumanWalkParams) to cope with their own deficits - the naive human
population, until the learned human of Phase 1 exists.

Each candidate walks with the zero-torque exo in several deficit variants (fresh; volitional drive ramping down after
5 s to a lower level; fatigue accumulating from the start), deficits in "capacity" mode (the person shares the load
within what is left). Cost per run as in indego_sim's tools/diag/tune_own_walk.py: time walked, counted only while
moving forward at >= vmin, + pw x forward distance (capped at vcap x time), averaged over runs.

    python scripts/tune_robust_walk.py out.jsonl [iters=30] [popsize=16] [T=30] [seeds=0,1] [subject=sci_t12_aisD]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

VARIANTS = [{"kind": "fresh"}, {"kind": "drive", "level": 0.7}, {"kind": "drive", "level": 0.5},
            {"kind": "fatigue", "rate": 30.0}]
VMIN, VCAP, PW = 0.15, 0.6, 1.0


def bounds():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "indego_sim", "tools",
                                    "diag"))
    from tune_own_walk import BOUNDS
    return BOUNDS


def run(job):
    walk, variant, seed, T, subject = job
    from indego_sim.config import HumanWalkParams, VolitionParams
    from indego_aan.deficits import Deficits, DriveProcess, Fatigue3CCr
    from indego_aan.scenario import make_walking_sim
    import indego_aan.scenario as sc
    orig = sc.HumanWalkParams.for_subject
    sc.HumanWalkParams.for_subject = classmethod(lambda cls, name: HumanWalkParams(**walk))
    try:
        sim, obs = make_walking_sim(seed=seed, name=subject)
    finally:
        sc.HumanWalkParams.for_subject = orig
    if variant["kind"] == "drive":
        lv = variant["level"]
        Deficits(sim, drive=DriveProcess(schedule=[(0, 1.0), (5.0, 1.0), (15.0, lv), (1e9, lv)]))
    elif variant["kind"] == "fatigue":
        Deficits(sim, fatigue=Fatigue3CCr(sim.human.mm.names, rate_scale=variant["rate"]))
    x0 = sim.d.subtree_com[sim.root][0]
    for _ in range(int(round(T / sim.ctrl_dt))):
        obs = sim.step(np.zeros(4), record=False)
        if sim.fell:
            break
    t = sim.d.time - sim.t0 if sim.fell else T
    return dict(kind=variant["kind"], t=round(t, 2), prog=round(float(sim.d.subtree_com[sim.root][0] - x0), 2))


def cost(runs):
    c = 0.0
    for r in runs:
        t_eff = min(r["t"], 2.0 + max(r["prog"], 0.0) / VMIN)
        c -= t_eff + PW * max(min(r["prog"], VCAP * r["t"]), 0.0) - 4.0 * max(-r["prog"], 0.0)
    return c / len(runs)


if __name__ == "__main__":
    import cma
    from indego_sim.config import HumanWalkParams
    out = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    pop = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    T = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    seeds = [int(s) for s in (sys.argv[5] if len(sys.argv) > 5 else "0,1").split(",")]
    subject = sys.argv[6] if len(sys.argv) > 6 else "sci_t12_aisD"
    B = bounds()
    start = HumanWalkParams.for_subject(subject).__dict__
    x0 = [float(np.clip((start[k] - lo) / (hi - lo), 0.02, 0.98)) for k, (lo, hi) in B.items()]

    def decode(x):
        w = dict(start)
        w.update({k: round(float(lo + (hi - lo) * xi), 4) for (k, (lo, hi)), xi in zip(B.items(), np.clip(x, 0, 1))})
        return w
    es = cma.CMAEvolutionStrategy(x0, 0.08, {"bounds": [0, 1], "popsize": pop, "seed": 5, "verbose": -9})
    best = (np.inf, None, None)
    from indego_aan.resources import single_thread_env, worker_count
    single_thread_env()
    with Pool(worker_count()) as pool, open(out, "w") as f:
        for it in range(iters):
            t0 = time.time()
            X = es.ask()
            walks = [decode(x) for x in X]
            jobs = [(w, v, s, T, subject) for w in walks for v in VARIANTS for s in seeds]
            res = pool.map(run, jobs)
            n = len(VARIANTS) * len(seeds)
            costs = []
            for i, w in enumerate(walks):
                runs = res[i * n:(i + 1) * n]
                costs.append(cost(runs))
                f.write(json.dumps(dict(it=it, cost=costs[-1], runs=runs, walk=w)) + "\n")
                if costs[-1] < best[0]:
                    best = (costs[-1], w, runs)
            f.flush()
            es.tell(X, costs)
            print(f"iter {it}: best {best[0]:.2f} {[(r['kind'][0], r['t'], r['prog']) for r in best[2]]} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    print(json.dumps(best[1]))
