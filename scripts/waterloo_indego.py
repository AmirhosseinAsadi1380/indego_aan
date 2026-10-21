"""Real people in the Indego vs the simulated person in the simulated Indego (Waterloo Indego dataset).

Data: Nasiri, Dinovitzer, Manohara & Arami, "Coordinated human-exoskeleton locomotion emerges from regulating
virtual energy", PLOS ONE 2025 (doi 10.1371/journal.pone.0292334); Figshare 10.6084/m9.figshare.24910953 (CC BY 4.0).
Experiment 1: 9 healthy adults on a treadmill at 0.60 / 0.75 / 0.85 m/s, without the exo, in the Indego with zero
applied torque ("passive") and with the virtual energy regulator ("ver"). The data stay outside this repository.

What it computes, per subject and speed:
  * exo encoders (hip-peak cycles): stride time, joint ranges, knee stance extension, hip swing flexion, per block;
    the paper's VER-vs-passive changes (reproduction check); VER commanded torque, power and work;
  * thigh IMU vs encoder: sagittal gyro gain and lag, accelerometer along the thigh;
  * markers (heel-strike cycles from the treadmill belts): hip and knee without the exo vs in the exo;
  * EMG (16 muscles): the paper's muscle effort (sum of cubed normalised activity per stride; normalised to the
    passive block, not MVC) and on/off timing per block.
It also applies the joint-angle realism test (hip and knee inside the in-exo mean +- 2 SD band for >= 90 % of the
cycle) to every simulated or recorded trial.
With --simulate it also runs the simulated person walking on their own in the zero-torque exo (the counterpart of
"passive"): healthy (indego_sim's tuned walker) and T12 AIS D (indego_aan's robust walker), and compares the same
quantities, incl. muscle excitations vs EMG timing.

    python scripts/waterloo_indego.py <data_dir> [--subjects 1-9] [--simulate] [--seeds 4] [--T 40]
        [--out results/waterloo] [--workers auto]

Any recorded trial in the same format (see `save_trial`) can be compared with --trial <npz> (e.g. a KINESIS run).
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from indego_aan import waterloo as W                     # noqa: E402
from indego_aan.resources import single_thread_env, worker_count   # noqa: E402

PAPER = {   # Nasiri et al. 2025, VER vs passive, mean +- SD over 9 people, at 0.60 / 0.75 / 0.85 m/s
    "effort_reduction_pct": ([14.4, 17.7, 13.1], [4.0, 6.2, 5.7]),
    "knee_stance_extension_gain_deg": ([3.9, 2.9, 4.1], [4.7, 5.1, 4.9]),
    "hip_swing_flexion_gain_deg": ([2.6, 2.5, 1.7], [3.2, 2.8, 3.6]),
    "median_muscle_reduction_pct": ([14.4, 17.7, 13.1], [4.0, 6.2, 5.7]),   # ours only; paper column = its effort
}
PHASE1_INSIDE_FRAC = 0.90       # Phase 1 joint-angle test: share of the cycle inside the in-exo +-2 SD band
SIM_PEOPLE = {"healthy": ("healthy", "tuned"), "t12_aisD": ("sci_t12_aisD", "robust")}


# ------------------------------------------------------------------------------------------------ exo encoders

def block_slices(t, blocks):
    out = {}
    for name, (a, b) in blocks.items():
        b = t[-1] if b is None else b
        sel = np.flatnonzero((t >= a) & (t < b))
        if sel.size:
            out[name] = slice(sel[0], sel[-1] + 1)
    return out


def encoder_features(q, qd, fs, gyr=None, acc=None, tau=None):
    """Gait features of one block from exo-format signals (q, qd: (n, 4) deg, deg/s in hip_l, knee_l, hip_r, knee_r
    order; gyr/acc: {"l": (n, 3), "r": (n, 3)}; tau: (n, 4) applied torque or None). Sides averaged."""
    per, curves = [], {}
    for side, (ih, ik) in (("l", (0, 1)), ("r", (2, 3))):
        st = W.good_strides(W.hip_peaks(q[:, ih], fs), fs)
        if len(st) < 5:
            continue
        hc, kc = W.normalise(q[:, ih], st), W.normalise(q[:, ik], st)
        f = W.joint_features(hc, kc)
        dur = np.diff(st, axis=1)[:, 0] / fs
        f.update(stride_time=float(dur.mean()), stride_time_cv=float(dur.std() / dur.mean()), n_strides=len(st))
        curves[f"hip_{side}"], curves[f"knee_{side}"] = hc.mean(0), kc.mean(0)
        if gyr is not None:
            g = _sagittal(gyr[side], qd[:, ih])
            r, lag, gain = W.sagittal_gyro_gain(g, qd[:, ih], fs)
            f.update(gyro_corr=r, gyro_lead_s=lag, gyro_gain=gain, gyro_sd=float(g.std()), hip_qd_sd=float(qd[:, ih].std()))
            th = np.cumsum(g - g.mean()) / fs
            th = th - W.lowpass(th, fs, 0.3)                       # thigh angle in the world, drift removed
            f["thigh_rom_gyro"] = float(np.mean([np.ptp(th[s:e]) for s, e in st]))
        if acc is not None:
            a = acc[side][:, int(np.argmax(np.abs(acc[side].mean(0))))]
            f.update(acc_axial_mean_g=float(abs(a.mean())), acc_axial_sd_g=float(a.std()))
        if tau is not None:
            for j, name in ((ih, "hip"), (ik, "knee")):
                P = tau[:, j] * np.radians(qd[:, j])
                pos = [np.sum(np.clip(P[s:e], 0, None)) / fs for s, e in st]
                neg = [np.sum(np.clip(P[s:e], None, 0)) / fs for s, e in st]
                f.update({f"{name}_tau_peak": float(np.mean([np.abs(tau[s:e, j]).max() for s, e in st])),
                          f"{name}_work_pos_J": float(np.mean(pos)), f"{name}_work_neg_J": float(np.mean(neg)),
                          f"{name}_power_W": float(P[st[0, 0]:st[-1, 1]].mean())})
                curves[f"tau_{name}_{side}"] = W.normalise(tau[:, j], st).mean(0)
        per.append(f)
    if not per:
        return None, curves
    feats = {k: float(np.mean([p[k] for p in per if k in p])) for k in per[0]}
    feats["cadence_spm"] = 120.0 / feats["stride_time"]
    return feats, curves


def _sagittal(g3, hip_qd):
    """The gyro axis (and sign) that follows the hip velocity best."""
    c = [np.corrcoef(g3[:, i], hip_qd)[0, 1] for i in range(3)]
    i = int(np.argmax(np.abs(c)))
    return g3[:, i] * np.sign(c[i])


# --------------------------------------------------------------------------------------------- one subject

def analyse_subject(args):
    root, sub = args
    out, curves = {"subject": sub}, {}
    for v, speed in W.SPEEDS.items():
        p = W.find_file(root, sub, "exo", v)
        if p is None:
            continue
        ex = W.load_exo(p)
        fs = ex["fs"]
        rv = {"speed": speed, "blocks": {}}
        for blk, sl in block_slices(ex["t"], W.BLOCKS).items():
            tau = ex["tau_cmd"][sl] if blk == "ver" else None          # commanded torque is applied only in VER
            imu = "gyr_thigh_l" in ex
            f, c = encoder_features(ex["q"][sl], ex["qd"][sl], fs,
                                    gyr={"l": ex["gyr_thigh_l"][sl], "r": ex["gyr_thigh_r"][sl]} if imu else None,
                                    acc={"l": ex["acc_thigh_l"][sl], "r": ex["acc_thigh_r"][sl]} if imu else None,
                                    tau=tau)
            if f:
                rv["blocks"][blk] = {"encoder": f}
                curves.update({f"s{sub}_v{v}_{blk}_enc_{k}": x for k, x in c.items()})
        b = rv["blocks"]
        if "passive" in b and "ver" in b:
            pe, ve = b["passive"]["encoder"], b["ver"]["encoder"]
            rv["paper_check"] = {"knee_stance_extension_gain_deg": pe["knee_stance_min"] - ve["knee_stance_min"],
                                 "hip_swing_flexion_gain_deg": ve["hip_max"] - pe["hip_max"]}
        vic = vicon_trial(root, sub, v, ex)
        if vic:
            res, c = vic
            for blk, r in res.pop("blocks").items():
                rv["blocks"].setdefault(blk, {}).update(r)
            rv.update(res)
            curves.update({f"s{sub}_v{v}_{k}": x for k, x in c.items()})
            e = rv["blocks"]
            if "effort" in e.get("passive", {}) and "effort" in e.get("ver", {}):
                rv.setdefault("paper_check", {})["effort_reduction_pct"] = \
                    100 * (1 - e["ver"]["effort"] / e["passive"]["effort"])
                lp, lv = e["passive"].get("emg_level", {}), e["ver"].get("emg_level", {})
                ratios = [lv[k] / lp[k] for k in lp if k in lv and lp[k] > 0]
                if ratios:                                   # robust to one artefact channel, unlike the cubes
                    rv["paper_check"]["median_muscle_reduction_pct"] = 100 * (1 - float(np.median(ratios)))
        out[f"v{v}"] = rv
    noexo = vicon_trial(root, sub, None, None)
    if noexo:
        res, c = noexo
        out["noexo"] = res
        curves.update({f"s{sub}_noexo_{k}": x for k, x in c.items()})
    return out, curves


def vicon_trial(root, sub, v, ex):
    """Markers + GRF + EMG of one recording, split into the exo blocks (synchronised on the knee angle) or, without
    the exo, into the three speeds."""
    pm = W.find_file(root, sub, "marker", v)
    if pm is None:
        return None
    mk = W.markers(pm)
    ang = {s: W.marker_angles(mk, s) for s in "RL"}
    pel = (mk["LASI"] + mk["RASI"]) / 2 - (mk["LPSI"] + mk["RPSI"]) / 2
    tilt = np.degrees(np.arctan2(pel[:, 2], pel[:, 1]))             # pelvis pitch in the lab (y forward, z up)
    res, curves = {"blocks": {}}, {}
    if ex is not None:
        off, r, r2 = W.sync_offset(ex["t"], ex["q"][:, 3], mk["t"], ang["R"]["knee"])
        res["sync"] = {"offset_s": off, "r": r, "r_next_best": r2}
        if not (r >= 0.6 and r - r2 >= 0.05):                     # also rejects NaN
            res["sync"]["rejected"] = True
            return res, curves
        windows = {k: (a - off, (ex["t"][-1] if b is None else b) - off) for k, (a, b) in W.BLOCKS.items()}
    else:
        windows = {f"v{k}": ab for k, ab in W.NOEXO_BLOCKS.items()}
    # heel strikes: treadmill belts, or the heel marker (Zeni et al. 2008) where the force data are missing
    pg = W.find_file(root, sub, "grf", v)
    hs = {}
    if pg is not None:
        rate, groups, _, grf = W.read_vicon_csv(pg)
        tg = W.vicon_time(grf, rate)
        for s, name in (("R", "Right - Force"), ("L", "Left - Force")):
            col = groups.index(name) + 2                              # Fx, Fy, Fz
            hs[s] = tg[W.heel_strikes(-grf[:, col], rate)]
        res["events"] = "grf"
    else:
        sac = (mk["LPSI"][:, 1] + mk["RPSI"][:, 1]) / 2
        from scipy.signal import find_peaks
        for s in "RL":
            x = np.nan_to_num(mk[s + "HEE"][:, 1] - sac)
            pk, _ = find_peaks(W.lowpass(x, 100.0, 6.0), distance=60, prominence=100.0)
            hs[s] = mk["t"][pk]
        res["events"] = "heel marker"
    pe = W.find_file(root, sub, "emg", v)
    env = None
    if pe is not None:
        rate, _, _, emg = W.read_vicon_csv(pe)
        te = W.vicon_time(emg, rate)
        raw = emg[:, 2:18]
        rms = np.sqrt(np.mean(raw ** 2, axis=0))
        valid = (rms > 1e-6) & (rms < 2e-3)                         # surface EMG: 1 uV - 2 mV RMS
        res["emg_valid_channels"] = int(valid.sum())
        if valid.any():
            e = W.emg_envelope(raw, rate)[::20]                     # 100 Hz
            e[:, ~valid] = np.nan
            env = (te[::20], e)
    tm = mk["t"]
    norm = None
    for blk, (a, b) in windows.items():
        r = {}
        for s in "RL":
            ev = hs[s][(hs[s] >= a) & (hs[s] < b)]
            idx = np.searchsorted(tm, ev)
            st = W.good_strides(idx, 100.0)
            if len(st) < 5:
                continue
            hc, kc = W.normalise(ang[s]["hip"], st), W.normalise(ang[s]["knee"], st)
            ok = ~np.isnan(hc).any(1) & ~np.isnan(kc).any(1)
            if ok.sum() < 5:
                continue
            curves[f"{blk}_mk_hip_{s}"], curves[f"{blk}_mk_knee_{s}"] = hc[ok].mean(0), kc[ok].mean(0)
            r.setdefault("markers_side", []).append({
                "hip_rom": float(np.ptp(hc[ok], 1).mean()), "knee_rom": float(np.ptp(kc[ok], 1).mean()),
                "hip_max": float(hc[ok].max(1).mean()), "hip_min": float(hc[ok].min(1).mean()),
                "knee_stance_min": float(kc[ok][:, 5:50].min(1).mean()), "knee_swing_max": float(kc[ok].max(1).mean()),
                "pelvis_tilt_rom": float(np.nanmean([np.ptp(tilt[a_:b_]) for a_, b_ in st])),
                "stride_time": float(np.diff(st, axis=1).mean() / 100.0)})
            if env is not None:
                ti, ei = env
                cols = slice(0, 8) if s == "R" else slice(8, 16)
                if norm is None:                                    # the first block: passive, or noexo low speed
                    sel = (ti >= a) & (ti < b)
                    norm = ei[sel].mean(0)
                idx_e = np.searchsorted(ti, ev)
                st_e = W.good_strides(idx_e, 100.0)
                cyc = W.normalise(ei[:, cols] / norm[cols], st_e)          # (strides, 101, 8)
                good = np.isfinite(cyc).all(axis=(0, 1))
                if not good.any() or len(st_e) < 5:
                    continue
                # the paper's effort (sum of cubed activity per stride), median stride: robust to artefact bursts
                per_stride = np.sum(np.mean(cyc[:, :, good] ** 3, axis=1), axis=1)
                r.setdefault("effort_side", []).append(float(np.median(per_stride)))
                for k, mname in enumerate(W.EMG_MUSCLES):
                    if good[k]:
                        curves[f"{blk}_emg_{mname}_{s}"] = cyc[:, :, k].mean(0)
                        r.setdefault("emg_level", {})[f"{mname}_{s}"] = float(np.median(cyc[:, :, k].mean(1)))
        if "markers_side" in r:
            ms = r.pop("markers_side")
            r["markers"] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        if "effort_side" in r:
            r["effort"] = float(np.mean(r.pop("effort_side")))
        res["blocks"][blk] = r
    return res, curves


# ----------------------------------------------------------------------------------------------- simulation

def simulate(args):
    """The simulated person walking on their own in the zero-torque exo; returns a trial dict (see save_trial)."""
    person, seed, T = args
    single_thread_env()
    from indego_aan.scenario import make_walking_sim
    name, human = SIM_PEOPLE[person]
    sim, obs = make_walking_sim(seed=seed, name=name, human=human, walk_at=0.5, log_rate_hz=1.0)
    rec = W.TrialRecorder(sim)
    for _ in range(int(round(T / sim.ctrl_dt))):
        obs = sim.step(np.zeros(4), record=False)
        rec(obs)
        if sim.fell:
            break
    return rec.arrays(person=person, seed=seed)


def save_trial(path, tr):
    """Trial format: fs; q, qd (n, 4) deg, deg/s (hip_l, knee_l, hip_r, knee_r); gyr_l/gyr_r, acc_l/acc_r (n, 3)
    deg/s and g; load (n, 2) vertical foot loads [N] (l, r); pelvis_pitch (n,) [deg]; exc (n, 16) muscle excitations in exc_order
    (EMG_MUSCLES right then left: "<muscle>_r"...); q_human (n, 4) the person's own joint angles [deg]; com_x [m]."""
    np.savez_compressed(path, **tr)


def analyse_trial(tr, t_start=10.0):
    fs = float(tr["fs"])
    i0 = int(t_start * fs)
    if len(tr["q"]) < i0 + 10 * fs:
        return None, {}
    q, qd = tr["q"][i0:], tr["qd"][i0:]
    f, c = encoder_features(q, qd, fs, gyr={"l": tr["gyr_l"][i0:], "r": tr["gyr_r"][i0:]},
                            acc={"l": tr["acc_l"][i0:], "r": tr["acc_r"][i0:]})
    if f is None:
        return None, {}
    com = tr["com_x"][i0:]
    f["speed"] = float((com[-1] - com[0]) / (len(com) / fs))
    # the person's knee vs the exo knee (cuff compliance)
    qh = tr["q_human"][i0:]
    sgn = np.sign(np.corrcoef(qh[:, 3], q[:, 3])[0, 1])
    f["knee_rom_human_over_exo"] = float(np.ptp(np.percentile(sgn * qh[:, 3], [1, 99])) /
                                         np.ptp(np.percentile(q[:, 3], [1, 99])))
    if "pelvis_pitch" in tr:
        pp = tr["pelvis_pitch"][i0:]
        st = W.good_strides(W.hip_peaks(q[:, 2], fs), fs)
        f["pelvis_tilt_rom"] = float(np.mean([np.ptp(pp[s:e]) for s, e in st]))
    curves = {f"enc_{k}": x for k, x in c.items()}
    order = [str(x) for x in tr["exc_order"]]
    exc = tr["exc"][i0:]
    k = np.ones(10) / 10                                             # 50 ms moving average, as the EMG envelope
    exc = np.stack([np.convolve(exc[:, j], k, mode="same") for j in range(exc.shape[1])], -1)
    for s, col in (("R", 1), ("L", 0)):
        st = W.good_strides(W.heel_strikes(tr["load"][i0:, col], fs), fs)
        if len(st) < 5:
            continue
        for mname in W.EMG_MUSCLES:
            j = order.index(f"{mname}_{s.lower()}")
            curves[f"emg_{mname}_{s}"] = W.normalise(exc[:, j], st).mean(0)
        f.setdefault("stride_time_hs", []).append(float(np.diff(st, axis=1).mean() / fs))
    if "stride_time_hs" in f:
        f["stride_time_hs"] = float(np.mean(f["stride_time_hs"]))
    return f, curves


# ----------------------------------------------------------------------------------------------- summaries

def group(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)} if v.size else None


def emg_timing(curve_a, curve_b):
    """Agreement of two cycle curves: Pearson r, shift [% cycle] that maximises r (+: b later), on/off agreement."""
    a, b = curve_a / curve_a.max(), curve_b / curve_b.max()
    r = float(np.corrcoef(a, b)[0, 1])
    shifts = np.arange(-25, 26)
    rs = [np.corrcoef(a, np.roll(b, -k))[0, 1] for k in shifts]
    return {"r": r, "best_shift_pct": int(shifts[int(np.argmax(rs))]), "r_shifted": float(max(rs)),
            "on_off_agree": float(np.mean(W.on_off(a) == W.on_off(b)))}


def summarise(subjects, curves, sims):
    out = {"paper_check": {}, "passive_vs_ver": {}, "noexo_vs_passive": {}, "sim_vs_real": {}}
    for v, speed in W.SPEEDS.items():
        rows = [s.get(f"v{v}", {}) for s in subjects]
        pc = {}
        for k, (pm, psd) in PAPER.items():
            g = group([r.get("paper_check", {}).get(k) for r in rows])
            if g:
                pc[k] = {"ours": g, "paper": {"mean": pm[v - 1], "sd": psd[v - 1]}}
        out["paper_check"][f"{speed:.2f}"] = pc
        blk = {}
        for b in W.BLOCKS:
            feats = [r.get("blocks", {}).get(b, {}).get("encoder") for r in rows]
            feats = [f for f in feats if f]
            if feats:
                blk[b] = {k: group([f.get(k) for f in feats]) for k in feats[0]}
        out["passive_vs_ver"][f"{speed:.2f}"] = blk
        mk = {}
        for name, getter in (("noexo", lambda s: s.get("noexo", {}).get("blocks", {}).get(f"v{v}", {}).get("markers")),
                             ("passive", lambda s: s.get(f"v{v}", {}).get("blocks", {}).get("passive", {}).get("markers")),
                             ("ver", lambda s: s.get(f"v{v}", {}).get("blocks", {}).get("ver", {}).get("markers"))):
            vals = [getter(s) for s in subjects]
            vals = [x for x in vals if x]
            if vals:
                mk[name] = {k: group([x[k] for x in vals]) for k in vals[0]}
        out["noexo_vs_passive"][f"{speed:.2f}"] = mk
    # EMG timing: in the exo (passive) vs without, per muscle (right leg), group mean curves
    out["emg_noexo_vs_passive"] = {}
    for v, speed in W.SPEEDS.items():
        res = {}
        for mname in W.EMG_MUSCLES:
            a = _mean_curve(curves, f"_noexo_v{v}_emg_{mname}_R")
            b = _mean_curve(curves, f"_v{v}_passive_emg_{mname}_R")
            if a is not None and b is not None:
                res[mname] = emg_timing(a, b)
        out["emg_noexo_vs_passive"][f"{speed:.2f}"] = res
    # simulation vs the real passive block at the nearest speed
    for person, trials in sims.items():
        fs_ = [f for f, _ in trials if f]
        if not fs_:
            continue
        sf = {k: group([f.get(k) for f in fs_]) for k in fs_[0] if not isinstance(fs_[0][k], list)}
        speed = sf["speed"]["mean"]
        v = min(W.SPEEDS, key=lambda k: abs(W.SPEEDS[k] - speed))
        real = dict(out["passive_vs_ver"][f"{W.SPEEDS[v]:.2f}"].get("passive", {}))
        mk_passive = out["noexo_vs_passive"][f"{W.SPEEDS[v]:.2f}"].get("passive", {})
        if "pelvis_tilt_rom" in mk_passive:
            real["pelvis_tilt_rom"] = mk_passive["pelvis_tilt_rom"]
        cmp = {}
        for k, g in sf.items():
            if k in real and real[k] and g:
                z = (g["mean"] - real[k]["mean"]) / real[k]["sd"] if real[k]["sd"] > 0 else None
                cmp[k] = {"sim": g, "real": real[k], "z": z}
        curves_cmp = {}
        for j in ("hip", "knee"):
            simc = np.mean([c[f"enc_{j}_{s}"] for _, c in trials for s in "lr" if f"enc_{j}_{s}" in c], axis=0)
            reals = _subject_curves(curves, f"_v{v}_passive_enc_{j}_")
            if reals is not None:
                mu, sd = reals.mean(0), reals.std(0, ddof=1)
                # offset-free: each curve minus its own mean (standing posture and strap fit shift the encoder
                # zero by person: real standing reads hip -8 +- 7 deg, knee +9 +- 8 deg; the sim reads 0)
                rc = reals - reals.mean(1, keepdims=True)
                sc = simc - simc.mean()
                mu_c, sd_c = rc.mean(0), rc.std(0, ddof=1)
                curves_cmp[j] = {"rmse_deg": float(np.sqrt(np.mean((simc - mu) ** 2))),
                                 "r": float(np.corrcoef(simc, mu)[0, 1]),
                                 "inside_2sd_frac": float(np.mean(np.abs(simc - mu) <= 2 * sd)),
                                 "shape_rmse_deg": float(np.sqrt(np.mean((sc - mu_c) ** 2))),
                                 "shape_inside_2sd_frac": float(np.mean(np.abs(sc - mu_c) <= 2 * sd_c)),
                                 "real_band_sd_deg": float(sd.mean()), "real_shape_band_sd_deg": float(sd_c.mean())}
        # joint-angle realism test: inside the in-exo +-2 SD band for >= 90 % of the cycle
        near = abs(W.SPEEDS[v] - speed) <= 0.1
        check = {"reference": f"healthy adults in the Indego, zero torque, {W.SPEEDS[v]:.2f} m/s",
                 "applies": bool(near), "threshold": PHASE1_INSIDE_FRAC}
        if near and all(j in curves_cmp for j in ("hip", "knee")):
            check.update({j: curves_cmp[j]["inside_2sd_frac"] for j in ("hip", "knee")})
            check["pass"] = all(curves_cmp[j]["inside_2sd_frac"] >= PHASE1_INSIDE_FRAC for j in ("hip", "knee"))
            check.update({f"{j}_offset_free": curves_cmp[j]["shape_inside_2sd_frac"] for j in ("hip", "knee")})
            check["pass_offset_free"] = all(curves_cmp[j]["shape_inside_2sd_frac"] >= PHASE1_INSIDE_FRAC
                                            for j in ("hip", "knee"))
        emg = {}
        for mname in W.EMG_MUSCLES:
            simc = [c[f"emg_{mname}_R"] for _, c in trials if f"emg_{mname}_R" in c]
            real_c = _mean_curve(curves, f"_v{v}_passive_emg_{mname}_R")
            if simc and real_c is not None:
                emg[mname] = emg_timing(real_c, np.mean(simc, 0))
        out["sim_vs_real"][person] = {"real_speed": W.SPEEDS[v], "sim_speed": sf["speed"], "n_trials": len(fs_),
                                      "falls": int(sum(1 for f, _ in trials if f is None)),
                                      "features": cmp, "curves": curves_cmp, "emg_timing": emg,
                                      "phase1_angle_check": check}
    return out


def _subject_curves(curves, tag):
    """(subjects, 101) curves whose key contains tag, sides averaged per subject."""
    per = {}
    for k, x in curves.items():
        if tag in k:
            per.setdefault(k.split("_")[0], []).append(x)
    return np.array([np.mean(v, 0) for v in per.values()]) if per else None


def _mean_curve(curves, tag):
    c = _subject_curves(curves, tag)
    if c is None:
        return None
    return np.mean([x / x.max() for x in c], 0)


def _colors(sims):
    palette = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
    return list(zip(sorted(sims), palette))


def figure(path, curves, sims, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 4, figsize=(15, 9))
    ph = np.linspace(0, 100, W.N_PHASE)
    for col, j in enumerate(("hip", "knee")):
        for row, v in enumerate(W.SPEEDS):
            a = ax[row, col]
            for blk, color in (("passive", "k"), ("ver", "tab:red")):
                c = _subject_curves(curves, f"_v{v}_{blk}_enc_{j}_")
                if c is not None:
                    mu, sd = c.mean(0), c.std(0, ddof=1)
                    a.plot(ph, mu, color, label=f"real {blk} (n={len(c)})")
                    a.fill_between(ph, mu - 2 * sd, mu + 2 * sd, color=color, alpha=0.12)
            for person, color in _colors(sims):
                tr = sims.get(person, [])
                cs = [c[f"enc_{j}_{s}"] for _, c in tr for s in "lr" if f"enc_{j}_{s}" in c]
                cmpv = summary["sim_vs_real"].get(person, {}).get("real_speed")
                if cs and cmpv == W.SPEEDS[v]:
                    a.plot(ph, np.mean(cs, 0), color=color, lw=2, label=f"sim {person}, zero torque")
            a.set_title(f"{j} (exo encoder), {W.SPEEDS[v]:.2f} m/s")
            a.set_xlabel("% cycle from hip-flexion peak")
            a.set_ylabel("deg")
            if row == 0:
                a.legend(fontsize=7)
    # EMG panels: six of the eight muscles, right leg
    for a, mname in zip([ax[0, 2], ax[0, 3], ax[1, 2], ax[1, 3], ax[2, 2], ax[2, 3]], ("GLU", "RF", "VAS", "BF", "TA", "SOL")):
        for tag, color, lab in (("_noexo_v2_emg_", "0.5", "real, no exo"), ("_v2_passive_emg_", "k", "real, passive exo")):
            c = _mean_curve(curves, f"{tag}{mname}_R")
            if c is not None:
                a.plot(ph, c / c.max(), color=color, label=lab)
        for person, color in _colors(sims):
            cs = [c[f"emg_{mname}_R"] for _, c in sims.get(person, []) if f"emg_{mname}_R" in c]
            if cs:
                mu = np.mean(cs, 0)
                a.plot(ph, mu / mu.max(), color=color, lw=2, label=f"sim {person} excitation")
        a.set_title(f"{mname} (right), heel-strike cycle, 0.75 m/s real")
        a.set_ylim(0, 1.1)
        if mname == "GLU":
            a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


# ----------------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--subjects", default="1-9")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--T", type=float, default=40.0)
    ap.add_argument("--trial", action="append", default=[],
                    help="recorded trial(s) to compare (npz, globs allowed); trials named <group>_s<seed>.npz are pooled "
                         "per group")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the saved analysis of the real data (summary.json, curves.npz) and only add trials; "
                         "writes summary_trials.json and waterloo_vs_trials.png")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "waterloo"))
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    single_thread_env()
    os.makedirs(a.out, exist_ok=True)
    lo, _, hi = a.subjects.partition("-")
    subs = list(range(int(lo), int(hi or lo) + 1)) if "," not in a.subjects else [int(x) for x in a.subjects.split(",")]
    subs = [s for s in subs if W.find_file(a.data_dir, s, "exo", 1)] if not a.reuse else []
    if not a.reuse:
        print("subjects with data:", subs, flush=True)
    jobs = [(a.data_dir, s) for s in subs]
    sim_jobs = [(p, s, a.T) for p in SIM_PEOPLE for s in range(a.seeds)] if a.simulate else []
    n = worker_count(n_jobs=len(jobs) + len(sim_jobs), requested=a.workers, per_worker_gb=1.2)
    with Pool(n) as pool:
        sim_async = pool.map_async(simulate, sim_jobs) if sim_jobs else None
        res = pool.map(analyse_subject, jobs)
        sim_trials = sim_async.get() if sim_async else []
    subjects = [r for r, _ in res]
    curves = {}
    for _, c in res:
        curves.update(c)
    if a.reuse:
        subjects = json.load(open(os.path.join(a.out, "summary.json")))["subjects"]
        curves = dict(np.load(os.path.join(a.out, "curves.npz")))
    for tr in sim_trials:
        save_trial(os.path.join(a.out, f"sim_{tr['person']}_s{tr['seed']}.npz"), tr)
    sims = {}
    sim_files = sorted(f for f in os.listdir(a.out) if f.startswith("sim_") and f.endswith(".npz"))
    for fn in sim_files:
        tr = dict(np.load(os.path.join(a.out, fn), allow_pickle=False))
        person = str(tr["person"])
        sims.setdefault(person, []).append(analyse_trial(tr))
    import glob as _glob
    import re as _re
    for pat in a.trial:
        for p in sorted(_glob.glob(pat)):
            tr = dict(np.load(p, allow_pickle=False))
            group = _re.sub(r"_s\d+$", "", os.path.splitext(os.path.basename(p))[0])
            sims.setdefault(group, []).append(analyse_trial(tr))
    summary = summarise(subjects, curves, sims)
    summary["source"] = ("Nasiri, Dinovitzer, Manohara & Arami, PLOS ONE 2025, doi 10.1371/journal.pone.0292334; "
                         "data: Figshare 10.6084/m9.figshare.24910953 (CC BY 4.0)")
    summary["subjects"] = subjects
    tag = "_trials" if a.reuse else ""
    with open(os.path.join(a.out, f"summary{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if not a.reuse:
        np.savez_compressed(os.path.join(a.out, "curves.npz"), **curves)
    figure(os.path.join(a.out, "waterloo_vs_trials.png" if a.reuse else "waterloo_vs_sim.png"), curves, sims, summary)
    print_report(summary)


def print_report(s):
    print("\n== paper check (VER vs passive; ours from the exo encoders / EMG normalised to the passive block)")
    for sp, pc in s["paper_check"].items():
        for k, d in pc.items():
            print(f"  {sp} m/s {k:34s} ours {d['ours']['mean']:6.1f} +- {d['ours']['sd']:4.1f} (n={d['ours']['n']})"
                  f"   paper {d['paper']['mean']:5.1f} +- {d['paper']['sd']:.1f}")
    print("\n== exo encoders, real (mean +- SD over people)")
    keys = ("stride_time", "cadence_spm", "hip_rom", "hip_max", "hip_min", "knee_rom", "knee_stance_min",
            "knee_swing_max", "gyro_gain", "gyro_lead_s")
    for sp, blk in s["passive_vs_ver"].items():
        for b, f in blk.items():
            print(f"  {sp} {b:13s} " + "  ".join(f"{k}={f[k]['mean']:.2f}+-{f[k]['sd']:.2f}" for k in keys if f.get(k)))
    print("\n== markers: without the exo vs in the exo")
    for sp, mk in s["noexo_vs_passive"].items():
        for b, f in mk.items():
            print(f"  {sp} {b:8s} " + "  ".join(f"{k}={v['mean']:.1f}+-{v['sd']:.1f}" for k, v in f.items()))
    print("\n== simulation (zero torque) vs real passive")
    for person, c in s["sim_vs_real"].items():
        print(f"  {person} (sim {c['sim_speed']['mean']:.2f} m/s) vs real at {c['real_speed']} m/s "
              f"({c['n_trials']} trials, {c['falls']} without data)")
        for k, d in c["features"].items():
            z = "" if d["z"] is None else f"z={d['z']:+.1f}"
            print(f"     {k:22s} sim {d['sim']['mean']:8.3f}   real {d['real']['mean']:8.3f} +- {d['real']['sd']:.3f}  {z}")
        for j, d in c["curves"].items():
            print(f"     {j} curve: RMSE {d['rmse_deg']:.1f} deg, r {d['r']:.2f}, inside real +-2SD {100 * d['inside_2sd_frac']:.0f} %"
                  f" | offset-free: RMSE {d['shape_rmse_deg']:.1f} deg, inside {100 * d['shape_inside_2sd_frac']:.0f} %"
                  f" (real band SD {d['real_band_sd_deg']:.1f} -> {d['real_shape_band_sd_deg']:.1f} deg)")
        for mname, d in c["emg_timing"].items():
            print(f"     EMG {mname:4s} r {d['r']:+.2f}  best shift {d['best_shift_pct']:+d} %  on/off agree {100 * d['on_off_agree']:.0f} %")
        pc = c["phase1_angle_check"]
        if "pass" in pc:
            print(f"     Phase 1 joint-angle test ({pc['reference']}): hip {100 * pc['hip']:.0f} %, knee "
                  f"{100 * pc['knee']:.0f} % of the cycle inside +-2 SD (need {100 * pc['threshold']:.0f} %) -> "
                  f"{'PASS' if pc['pass'] else 'FAIL'}; offset-free hip {100 * pc['hip_offset_free']:.0f} %, knee "
                  f"{100 * pc['knee_offset_free']:.0f} % -> {'PASS' if pc['pass_offset_free'] else 'FAIL'}")
        else:
            print("     Phase 1 joint-angle test: no in-exo reference within 0.1 m/s of this speed")


if __name__ == "__main__":
    main()
