"""The personal AAN on real Indego logs (Waterloo dataset, 9 healthy adults, zero applied torque), open loop.

For each person and speed, the first half of the first zero-torque block (20-70 s) plays session B: it gives the
person's fresh reference (exo-encoder hip and knee curves over the stride, their step-to-step SD, the stride time).
The controller then reads the encoders of the held-out walking (70-120 s and the zero-torque block after VER) and
computes what it would command. Nothing is applied (the data are recorded), so this checks the parts that do not
depend on the assistance: does the phase estimate follow the real gait, and does a fresh healthy person stay inside
the controller's dead band (assistance near zero, "minimal assistance when fresh")? It also reports the real
step-to-step variability the dead band is built on, against the simulated people's.

    python scripts/paan_real_replay.py <waterloo data dir> [--out results/paan_real_replay.json]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from indego_aan import waterloo as W                             # noqa: E402
from indego_aan.controllers import PersonalAAN                   # noqa: E402

N = 101


def cycles(q, fs, a, b):
    """Right-hip-peak to right-hip-peak strides inside [a, b) samples: (start, end) pairs."""
    pk = W.hip_peaks(q[:, 2], fs)
    pk = pk[(pk >= a) & (pk < b)]
    return W.good_strides(pk, fs)


def reference(q, fs, st):
    """PersonalAAN reference (radians) from the strides st, phase 0 = right hip-flexion peak."""
    c = W.normalise(np.radians(q), st)                          # (strides, 101, 4)
    stride = float(np.mean(np.diff(st, axis=1))) / fs
    return dict(mean=c.mean(0), sd=c.std(0), step_time=stride / 2.0), c


def replay(ex, ref, a, b):
    """Run the controller over samples [a, b): phase, gains, torque per sample."""
    fs = ex["fs"]
    ctrl = PersonalAAN(ref, dt=1.0 / fs)
    q, qd = np.radians(ex["q"][a:b]), np.radians(ex["qd"][a:b])
    out = {"phase": [], "k": [], "tau": []}
    for i in range(b - a):
        tau = ctrl({"q": q[i], "qd": qd[i]})
        out["phase"].append(ctrl.phase)
        out["k"].append(ctrl.k.copy())
        out["tau"].append(tau)
    return {k: np.array(v, float) for k, v in out.items()}


def true_phase(n, st_abs, a):
    """Phase 0..1 between consecutive right hip peaks (NaN outside strides), for samples a..a+n."""
    ph = np.full(n, np.nan)
    for s, e in st_abs:
        i0, i1 = max(s - a, 0), min(e - a, n)
        if i1 > i0:
            ph[i0:i1] = (np.arange(i0, i1) + a - s) / (e - s)
    return ph


def analyse(ex, ref, a, b):
    fs = ex["fs"]
    r = replay(ex, ref, a, b)
    st = cycles(ex["q"], fs, a, b)
    tp = true_phase(b - a, st, a)
    ok = np.isfinite(tp)
    err = (r["phase"][ok] - tp[ok] + 0.5) % 1.0 - 0.5                # circular phase error, fraction of stride
    settle = int(5 * fs)                                             # the first seconds: the gains start at 0 anyway
    tau = r["tau"][settle:]
    qd = np.radians(ex["qd"][a:b][settle:])
    power = tau * qd
    return {
        "phase_err_median_pct": float(100 * np.median(np.abs(err))),
        "phase_err_p95_pct": float(100 * np.percentile(np.abs(err), 95)),
        "mean_abs_tau_Nm": float(np.mean(np.abs(tau))),
        "peak_tau_Nm_p99": float(np.percentile(np.abs(tau), 99)),
        "exo_pos_work_W": float(np.mean(np.clip(power, 0, None).sum(1))),
        "frac_time_k_above_0.1": float(np.mean((r["k"][settle:] > 0.1).any(axis=1))),
        "k_mean": r["k"][settle:].mean(0).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "paan_real_replay.json"))
    a = ap.parse_args()
    rows = []
    for sub in range(1, 10):
        for v, speed in W.SPEEDS.items():
            p = W.find_file(a.data_dir, sub, "exo", v)
            if p is None:
                continue
            ex = W.load_exo(p)
            fs, t = ex["fs"], ex["t"]
            i = lambda s: int(np.searchsorted(t, s))                  # noqa: E731
            st = cycles(ex["q"], fs, i(20), i(70))
            if len(st) < 10:
                continue
            ref, c = reference(ex["q"], fs, st)
            sd_deg = np.degrees(ref["sd"]).mean(0)                    # mean step-to-step SD per channel over the stride
            row = {"subject": sub, "speed": speed, "n_ref_strides": len(st), "stride_s": 2 * ref["step_time"],
                   "sd_deg": sd_deg.tolist()}
            row["held_out_zero_torque"] = analyse(ex, ref, i(70), i(120))
            if t[-1] > 270:
                row["zero_torque_after_ver"] = analyse(ex, ref, i(260), len(t))
            row["during_ver"] = analyse(ex, ref, i(140), i(240))
            rows.append(row)
            h = row["held_out_zero_torque"]
            print(f"s{sub} {speed:.2f} m/s: SD hip/knee {sd_deg[[0, 2]].mean():.1f}/{sd_deg[[1, 3]].mean():.1f} deg | "
                  f"held-out fresh: phase err {h['phase_err_median_pct']:.1f} % (p95 {h['phase_err_p95_pct']:.1f}), "
                  f"|tau| {h['mean_abs_tau_Nm']:.2f} N*m, work {h['exo_pos_work_W']:.2f} W, gains on "
                  f"{100 * h['frac_time_k_above_0.1']:.0f} % | during VER |tau| {row['during_ver']['mean_abs_tau_Nm']:.2f}",
                  flush=True)
    summary = {}
    for key in ("held_out_zero_torque", "zero_torque_after_ver", "during_ver"):
        vals = [r[key] for r in rows if key in r]
        summary[key] = {m: float(np.mean([v[m] for v in vals])) for m in
                        ("phase_err_median_pct", "phase_err_p95_pct", "mean_abs_tau_Nm", "peak_tau_Nm_p99",
                         "exo_pos_work_W", "frac_time_k_above_0.1")}
    summary["sd_deg_mean"] = np.mean([r["sd_deg"] for r in rows], axis=0).tolist()
    json.dump({"rows": rows, "summary": summary}, open(a.out, "w"), indent=1)
    print("\nmean over people and speeds:")
    for key, s in summary.items():
        print(f"  {key}: {s}")


if __name__ == "__main__":
    main()
