"""Phase 2 calibration targets from DUO-GAIT (Zhou et al. 2023, Sci Data 10:543; Zenodo 10.5281/zenodo.7415758):
16 healthy adults, 6-min overground walks before and after a sit-to-stand fatigue protocol (30 % body-mass vest, to
exhaustion, Borg > 15.5). Single-task walks only (the simulated person has no cognitive task). Per subject, the
change from the fresh to the fatigued walk in each aggregate gait parameter; reported as the mean relative change
across subjects, its SD and a paired t statistic.

Download processed.zip from Zenodo into data/duo_gait/ and unzip it there (the data are not redistributed here).

    python scripts/duo_gait.py -> results/duo_gait_targets.json
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data", "duo_gait", "processed")
PARAMS = ["speed_avg", "stride_times_avg", "stride_lengths_avg", "cadence_avg", "stance_ratios_avg",
          "stride_times_CV", "stride_lengths_CV", "speed_CV", "stride_times_SI"]


def load(cond):
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, cond, "sub_*", "aggregate_params.csv"))):
        sub = os.path.basename(os.path.dirname(f))
        cols, vals = (line.strip().split(",") for line in open(f).readlines()[:2])
        out[sub] = {c: float(v) for c, v in zip(cols, vals)}
    return out


def targets():
    pre, post = load("OG_st_control"), load("OG_st_fatigue")
    subs = sorted(set(pre) & set(post))
    res = {"n_subjects": len(subs)}
    for p in PARAMS:
        a = np.array([pre[s][p] for s in subs])
        b = np.array([post[s][p] for s in subs])
        rel = (b - a) / a
        res[p] = {"fresh": float(a.mean()), "fatigued": float(b.mean()), "rel_change_mean": float(rel.mean()),
                  "rel_change_sd": float(rel.std(ddof=1)),
                  "t_paired": float((b - a).mean() / ((b - a).std(ddof=1) / np.sqrt(len(subs))))}
    return res


if __name__ == "__main__":
    res = targets()
    out = os.path.join(ROOT, "results", "duo_gait_targets.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"DUO-GAIT single-task, {res['n_subjects']} subjects, fatigued vs fresh:")
    for p in PARAMS:
        r = res[p]
        print(f"  {p:20s} {r['fresh']:8.3f} -> {r['fatigued']:8.3f}   {100 * r['rel_change_mean']:+6.1f} % "
              f"(SD {100 * r['rel_change_sd']:.1f} %, t {r['t_paired']:+.2f})")
