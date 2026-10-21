"""Phase 5: low drive or fatigue? From Indego sensors only, through the deployable personal AAN.

scripts/deficit_estimate.py showed that the AAN's adapted gains track how much capacity the person has lost, but as
one number. Two physiological signatures could separate the causes:
  * time course - a drop in volitional drive is abrupt (seconds), fatigue builds up over many strides;
  * pattern - low drive weakens every muscle, fatigue mainly the hard-working extensors, so the hip and knee gains
    and the stride timing may respond differently.
Runs the AAN under several levels of each deficit, builds windowed features from the logged signals (gain levels,
their hip/knee ratio, slopes over the last 2 and 8 s, stride-time variability), and classifies drive vs fatigue on
samples where capacity has dropped below 0.9, with leave-one-seed-out and leave-one-level-out tests.

    python scripts/deficit_type.py [seeds=0,1,2,3] -> results/deficit_type.json
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DRIVE = [0.5, 0.6, 0.7]
FATIGUE = [15.0, 30.0, 60.0]
PREFATIGUE = [0.3, 0.5]
CONDITIONS = [("drive", v) for v in DRIVE] + [("fatigue", v) for v in FATIGUE] + [("prefatigue", v) for v in PREFATIGUE]
HZ = 20.0                                     # deficit_estimate logs at 20 Hz


def features(rows):
    """Per sample: [k_hip, k_knee, hip/knee ratio, slope of k_hip and k_knee over 2 s and 8 s, stride-time SD over
    the last 5 s, tau_abs]."""
    t = np.array([r["t"] for r in rows])
    kh = np.array([r["k_hip"] for r in rows])
    kk = np.array([r["k_knee"] for r in rows])
    st = np.array([r["stride_time"] for r in rows])
    ta = np.array([r["tau_abs"] for r in rows])
    out = []
    for i in range(len(rows)):
        f = [kh[i], kk[i], kh[i] / (kk[i] + 1e-3)]
        for win in (2.0, 8.0):
            j = max(0, i - int(win * HZ))
            dt = max(t[i] - t[j], 1e-3)
            f += [(kh[i] - kh[j]) / dt, (kk[i] - kk[j]) / dt]
        j = max(0, i - int(5.0 * HZ))
        s = st[j:i + 1]
        s = s[np.isfinite(s)]
        f += [float(np.std(s)) if len(s) > 2 else 0.0, ta[i]]
        out.append(f)
    return np.array(out)


def logistic_fit(X, y, l2=1e-2, iters=500):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):                                  # Newton steps (small problem)
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w, mu, sd


def predict(model, X):
    w, mu, sd = model
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    return 1 / (1 + np.exp(-Z @ w))


def samples(data, keep):
    """(X, y = 1 for drive, cause label) over samples whose capacity is below 0.9."""
    X, y = [], []
    for kind, level, seed, fell, rows in data:
        if not keep(kind, level, seed) or not rows:
            continue
        F = features(rows)
        cap = np.array([r["cap"] for r in rows])
        sel = cap < 0.9
        X.append(F[sel])
        y.append(np.full(sel.sum(), 1.0 if kind == "drive" else 0.0))
    return (np.concatenate(X), np.concatenate(y)) if X else (np.zeros((0, 9)), np.zeros(0))


if __name__ == "__main__":
    from deficit_estimate import run
    from indego_aan.resources import single_thread_env, worker_count
    seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3").split(",")]
    jobs = [(k, v, s) for k, v in CONDITIONS for s in seeds]
    single_thread_env()
    with Pool(worker_count(len(jobs))) as pool:
        res = pool.map(run, jobs)
    data = [(k, v, s, fell, [r for r in rows if r["t"] > 3.0 and np.isfinite(r["stride_time"])])
            for k, v, s, fell, rows in res]
    out = {"conditions": [f"{k} {v:g}" for k, v in CONDITIONS], "falls": {}}
    for k, v, s, fell, rows in data:
        out["falls"].setdefault(f"{k} {v:g}", 0)
        out["falls"][f"{k} {v:g}"] += int(fell)
    # leave one seed out
    acc = []
    for s in seeds:
        Xtr, ytr = samples(data, lambda k, v, sd: sd != s)
        Xte, yte = samples(data, lambda k, v, sd: sd == s)
        if len(yte) and len(set(ytr)) == 2:
            acc.append(float(np.mean((predict(logistic_fit(Xtr, ytr), Xte) > 0.5) == yte)))
    out["loso_accuracy"] = acc
    # leave one level out (a deficit size never seen in training)
    lolo = {}
    for k, v in CONDITIONS:
        Xtr, ytr = samples(data, lambda kk, vv, sd: (kk, vv) != (k, v))
        Xte, yte = samples(data, lambda kk, vv, sd: (kk, vv) == (k, v))
        if len(yte) and len(set(ytr)) == 2:
            lolo[f"{k} {v:g}"] = float(np.mean((predict(logistic_fit(Xtr, ytr), Xte) > 0.5) == yte))
    out["leave_one_level_out_accuracy"] = lolo
    # which features carry it: single-feature accuracy (leave one seed out)
    names = ["k_hip", "k_knee", "hip/knee", "dk_hip 2s", "dk_knee 2s", "dk_hip 8s", "dk_knee 8s", "stride SD 5s", "|tau|"]
    single = {}
    for i, n in enumerate(names):
        a = []
        for s in seeds:
            Xtr, ytr = samples(data, lambda k, v, sd: sd != s)
            Xte, yte = samples(data, lambda k, v, sd: sd == s)
            if len(yte) and len(set(ytr)) == 2:
                a.append(float(np.mean((predict(logistic_fit(Xtr[:, [i]], ytr), Xte[:, [i]]) > 0.5) == yte)))
        single[n] = float(np.mean(a)) if a else None
    out["single_feature_accuracy"] = single
    print("falls:", out["falls"])
    print(f"drive vs fatigue, leave-one-seed-out accuracy {np.mean(acc):.2f} ({', '.join(f'{x:.2f}' for x in acc)})")
    print("leave-one-level-out: " + ", ".join(f"{k} {x:.2f}" for k, x in lolo.items()))
    print("single features: " + ", ".join(f"{k} {x:.2f}" for k, x in single.items() if x is not None))
    json.dump(out, open(os.path.join(ROOT, "results", "deficit_type.json"), "w"), indent=1)
