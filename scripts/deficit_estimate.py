"""Phase 5 groundwork: can the deployable personal AAN tell how much the person has lost, from Indego sensors only?

The AAN's assistance gains rise when the tracking errors grow beyond the person's fresh variability, so they may
already be an interpretable estimate of the deficit. Runs the AAN (no privileged input) under each deficit, logs at
20 Hz sensor-side features - the mean hip and knee gains, the AAN torque, and the stride time from its phase
estimate - next to the true capacity (drive x (1 - hip/knee fatigue)), and reports (a) how the gains follow
1 - capacity, (b) a linear capacity estimate from the features, fitted on all seeds but one and tested on the
held-out seed (leave-one-seed-out).

    python scripts/deficit_estimate.py [seeds=0,1,2,3] -> results/deficit_estimate.json, results/deficit_estimate.png
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

DEFICITS = [("fresh", None), ("drive", 0.6), ("drive", 0.4), ("fatigue", 30.0), ("prefatigue", 0.5)]
T = 30.0
EVERY = 5                      # decisions (100 Hz) per logged sample -> 20 Hz
FEATURES = ["k_hip", "k_knee", "tau_abs", "stride_time"]


def run(job):
    kind, level, seed = job
    from indego_aan.env import env_from_config
    env = env_from_config({"base": "paan"}, deficit={"kind": kind, "level": level}, episode_s=T)
    env.reset(options={"sim_seed": seed})
    c = env.base_ctrl
    rows, tau_ema, last_wrap, stride, ph_prev, n = [], 0.0, None, np.nan, None, 0
    while True:
        _, _, term, trunc, info = env.step(np.zeros(4))
        n += 1
        tau_ema += 0.02 * (float(np.mean(np.abs(info["tau_base"]))) - tau_ema)      # ~0.5 s average
        ph = c.phase
        if ph is not None and ph_prev is not None and ph < ph_prev - 0.5:          # phase wrapped: a new stride
            if last_wrap is not None:
                stride = env.t - last_wrap
            last_wrap = env.t
        ph_prev = ph
        if n % EVERY == 0:
            fat = env.defi.fatigue.group_mean() if env.defi.fatigue is not None else {}
            rows.append(dict(t=env.t, cap=env.cap, drive=env.defi.drive_now,
                             fatigue=float(np.mean([fat.get(g, 0.0) for g in ("hip_extensors", "hip_flexors",
                                                                              "knee_extensors", "knee_flexors")])),
                             k_hip=float(np.mean(c.k[[0, 2]])), k_knee=float(np.mean(c.k[[1, 3]])),
                             tau_abs=tau_ema, stride_time=stride))
        if term or trunc:
            return kind, level, seed, bool(term), rows


def design(rows):
    X = np.array([[r[f] for f in FEATURES] for r in rows], float)
    return np.column_stack([np.ones(len(X)), X])


if __name__ == "__main__":
    from indego_aan.resources import single_thread_env, worker_count
    seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3").split(",")]
    jobs = [(k, v, s) for k, v in DEFICITS for s in seeds]
    single_thread_env()
    with Pool(worker_count(len(jobs))) as pool:
        res = pool.map(run, jobs)
    # samples once walking is established (stride time known)
    data = [(k, v, s, fell, [r for r in rows if r["t"] > 3.0 and np.isfinite(r["stride_time"])])
            for k, v, s, fell, rows in res]
    summary = {"conditions": [], "features": FEATURES}
    print(f"{'condition':15s} {'falls':>5s} {'capacity':>9s} {'k_hip':>6s} {'k_knee':>7s} {'|tau|':>6s} {'stride':>7s}"
          "   (last 10 s means)")
    for k, v in DEFICITS:
        g = [d for d in data if d[0] == k and d[1] == v]
        last = [r for d in g for r in d[4] if r["t"] > d[4][-1]["t"] - 10.0] if g else []
        m = {f: float(np.mean([r[f] for r in last])) for f in ["cap"] + FEATURES} if last else {}
        name = k if v is None else f"{k} {v:g}"
        summary["conditions"].append(dict(condition=name, falls=sum(d[3] for d in g), **m))
        print(f"{name:15s} {sum(d[3] for d in g):>3d}/{len(g)} {m.get('cap', np.nan):9.2f} {m.get('k_hip', np.nan):6.2f} "
              f"{m.get('k_knee', np.nan):7.2f} {m.get('tau_abs', np.nan):6.1f} {m.get('stride_time', np.nan):7.3f}")
    allr = [r for d in data for r in d[4]]
    loss = np.array([1.0 - r["cap"] for r in allr])
    corr = {f: float(np.corrcoef([r[f] for r in allr], loss)[0, 1]) for f in FEATURES}
    summary["corr_with_lost_capacity"] = corr
    print("correlation with lost capacity (1 - capacity), all samples: "
          + ", ".join(f"{f} {c:+.2f}" for f, c in corr.items()))
    # leave-one-seed-out linear estimate of the capacity
    errs, r2s = [], []
    for s in seeds:
        tr = [r for d in data if d[2] != s for r in d[4]]
        te = [r for d in data if d[2] == s for r in d[4]]
        w, *_ = np.linalg.lstsq(design(tr), np.array([r["cap"] for r in tr]), rcond=None)
        pred = design(te) @ w
        y = np.array([r["cap"] for r in te])
        errs.append(float(np.sqrt(np.mean((pred - y) ** 2))))
        r2s.append(float(1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)))
    summary.update(loso_rmse=errs, loso_r2=r2s)
    print(f"linear capacity estimate, leave-one-seed-out: RMSE {np.mean(errs):.3f} (capacity units), R^2 {np.mean(r2s):.2f}")
    # leave-one-deficit-out: fitted without a whole deficit type, tested on it (does it read the gait, or the condition?)
    loco = {}
    for k, v in DEFICITS:
        if k == "fresh":
            continue
        tr = [r for d in data if (d[0], d[1]) != (k, v) for r in d[4]]
        te = [r for d in data if (d[0], d[1]) == (k, v) for r in d[4] if r["t"] > d[4][-1]["t"] - 10.0]
        w, *_ = np.linalg.lstsq(design(tr), np.array([r["cap"] for r in tr]), rcond=None)
        pred, y = design(te) @ w, np.array([r["cap"] for r in te])
        loco[k if v is None else f"{k} {v:g}"] = dict(true=float(y.mean()), estimate=float(pred.mean()),
                                                      rmse=float(np.sqrt(np.mean((pred - y) ** 2))))
    summary["leave_one_deficit_out_last10s"] = loco
    print("leave one deficit type out (last 10 s): " + "; ".join(
        f"{n} true {x['true']:.2f} est {x['estimate']:.2f}" for n, x in loco.items()))
    # time course: estimate (fitted without the plotted seed) vs truth, and the delay after a drive drop
    s0 = seeds[0]
    w, *_ = np.linalg.lstsq(design([r for d in data if d[2] != s0 for r in d[4]]),
                            np.array([r["cap"] for d in data if d[2] != s0 for r in d[4]]), rcond=None)
    delays = []
    for k, v, sd, fell, rows in data:
        if k != "drive" or not rows:
            continue
        t = np.array([r["t"] for r in rows])
        est = design(rows) @ w
        cap = np.array([r["cap"] for r in rows])
        target = cap[0] + 0.63 * (cap[-1] - cap[0])            # 63 % of the way to the final capacity
        t_true = t[np.argmax(cap <= target)]
        hit = np.flatnonzero(est <= target)
        hit = hit[t[hit] >= 10.0]
        if len(hit):
            delays.append(float(t[hit[0]] - t_true))
    summary["drive_drop_delay_s"] = delays
    if delays:
        print(f"after a drive drop the estimate reaches 63 % of the change {np.mean(delays):.1f} s after the truth "
              f"(range {min(delays):.1f} to {max(delays):.1f} s)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(DEFICITS), figsize=(3.2 * len(DEFICITS), 3.0), sharey=True)
    for ax, (k, v) in zip(axes, DEFICITS):
        d = [x for x in data if x[0] == k and x[1] == v and x[2] == s0][0]
        t = np.array([r["t"] for r in d[4]])
        ax.plot(t, [r["cap"] for r in d[4]], color="#1f2937", lw=2, label="true capacity")
        ax.plot(t, design(d[4]) @ w, color="#2563eb", lw=1.2, label="estimate (sensors)")
        ax.set_title(k if v is None else f"{k} {v:g}", fontsize=10)
        ax.set_xlabel("time (s)")
        ax.set_ylim(0.0, 1.15)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("capacity (1 = fresh)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Deficit estimate from the personal AAN's gains, torque and stride time (seed {s0}, held out)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "deficit_estimate.png"), dpi=130)
    json.dump(summary, open(os.path.join(ROOT, "results", "deficit_estimate.json"), "w"), indent=1)
