"""The person's fresh gait reference, as lab session B would record it: zero-torque walking, exo encoder angles
(hip_l, knee_l, hip_r, knee_r) over the gait cycle - mean and step-to-step SD - and the step time; for the teacher's
gait reward (sim only) also the human hip, knee and ankle angles (mean_h, sd_h; channels in chan_h).

    python scripts/fresh_reference.py [seeds=0,1,2,3] [T=30] [human=robust] -> results/reference_<subject>_<human>.npz
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SUBJECT = "sci_t12_aisD"


def run(args):
    seed, T, human = args
    from indego_aan.rollout import run_episode
    from indego_aan.scenario import make_walking_sim
    sim, obs = make_walking_sim(seed=seed, human=human)
    return run_episode(sim, lambda o: np.zeros(4), T, obs=obs)


def reference_path(human="robust", subject=SUBJECT):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"reference_{subject}_{human}.npz")


if __name__ == "__main__":
    from indego_aan.metrics import GaitBaseline
    seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3").split(",")]
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    human = sys.argv[3] if len(sys.argv) > 3 else "robust"
    from indego_aan.resources import single_thread_env, worker_count
    single_thread_env()
    with Pool(worker_count(len(seeds))) as pool:
        recs = pool.map(run, [(s, T, human) for s in seeds])
    assert not any(r["fell"] for r in recs), "the fresh person fell"
    b = GaitBaseline.from_records(recs, t0=3.0, key="q_exo")
    h = GaitBaseline.from_records(recs, t0=3.0, key="q")
    out = reference_path(human)
    np.savez(out, mean=b.mean, sd=b.sd, step_time=b.step_time, speed=b.speed, n_cycles=b.n_cycles,
             mean_h=h.mean, sd_h=h.sd, chan_h=np.array(h.chan))
    print(f"{b.n_cycles} cycles, step {b.step_time:.3f} s, speed {b.speed:.2f} m/s; "
          f"hip range {np.degrees(np.ptp(b.mean[:, 2])):.0f} deg, knee range {np.degrees(np.ptp(b.mean[:, 3])):.0f} deg, "
          f"mean SD {np.degrees(b.sd.mean()):.1f} deg -> {os.path.relpath(out)}")
