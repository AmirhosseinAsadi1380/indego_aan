"""Phase 0: how many exo control steps (200 Hz) per second the full-fidelity sim delivers on this machine.

Runs the person-walks-on-their-own scenario (T12 AIS D, zero-torque exo, sagittal model) with logging reduced,
one process per core, and reports steps/s per process and in total plus the time 10^8 steps would take.

    python scripts/throughput.py [processes=all cores] [seconds of sim per process=20]
"""
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def run(args):
    seed, T = args
    from indego_aan.scenario import make_walking_sim
    import numpy as np
    sim, obs = make_walking_sim(seed=seed)
    n = int(round(T / sim.ctrl_dt))
    t0 = time.perf_counter()
    for _ in range(n):
        obs = sim.step(np.zeros(4), record=False)
        if sim.fell:
            break
    return n, time.perf_counter() - t0


if __name__ == "__main__":
    from indego_aan.resources import single_thread_env, worker_count
    single_thread_env()
    P = worker_count(requested=int(sys.argv[1]) if len(sys.argv) > 1 else None)
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    n1, w1 = run((0, T))
    single = n1 / w1
    with Pool(P) as pool:
        res = pool.map(run, [(s, T) for s in range(P)])
    total = sum(n / w for n, w in res)              # stepping only (model builds excluded)
    print(f"single process: {single:.0f} control steps/s ({single / 200:.2f} x real time)")
    print(f"{P} processes: {total:.0f} control steps/s in total ({total / P:.0f} per process)")
    print(f"10^8 steps: {1e8 / total / 3600:.1f} h on this machine; 10^8 steps at the single-process rate: "
          f"{1e8 / single / 86400:.1f} days")
