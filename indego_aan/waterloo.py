"""Reader and gait analysis for the Waterloo Indego dataset (Nasiri, Dinovitzer, Manohara & Arami, PLOS ONE 2025;
Figshare 10.6084/m9.figshare.24910953, CC BY 4.0).

Experiment 1 (subjects 1-9, healthy young adults) walked on a split-belt treadmill at 0.60 / 0.75 / 0.85 m/s
(files v1 / v2 / v3) without the exo (one "noexo" recording, low -> mid -> high speed, about 2 min each) and in the
Indego Explorer (one 5-min recording per speed: zero applied torque 0-120 s, the virtual energy regulator (VER)
120-240 s, zero torque again to the end; the exo log's `ctrl_en`). Rates: exo log 200 Hz, markers 100 Hz, IMU 200 Hz,
EMG 2000 Hz, GRF 1000 Hz. The exo log and the Vicon files (markers, GRF, EMG, IMU) have separate clocks;
`sync_offset` aligns them on the knee angle.

What the exo log holds (checked on subject 1):
  Q* joint angles [deg] (flexion +), W* velocities [deg/s] (a clean derivative of Q), E* VER virtual energy,
  GY*/AC* thigh IMUs: gyro [deg/s], sagittal axis z (+ on the right, - on the left thigh), accelerometer [g], y along
  the thigh. F* is the VER law's COMMANDED torque [N*m] (no torque sensor): it fits F = sign(W) (a + b E) with
  R^2 0.95-0.96 in BOTH conditions, so in the zero-torque blocks it is computed but never applied - ignore it there.

Subject 9's exo logs are the controller's raw workspace, named sub9_exo_v2/v3/v4: they match the mocap trials
v1/v2/v3 (knee-angle cross-correlation r 0.88 / 0.89 / 0.71 against <= 0.37 for the other trials), `ctrl_en` is
column 1 of `switches` (0 -> 1 at 120 s, back at 240 s) and there are no thigh-IMU channels. Subjects 2 and 8 ("no
grf") have their mocap/EMG in a different .mat layout and no no-exo recording: only their exo logs are read.

The data are not redistributed with this project: pass the folder that holds the unpacked subject folders.
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, Optional

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, find_peaks, correlate, correlation_lags

SPEEDS = {1: 0.60, 2: 0.75, 3: 0.85}
JOINTS = ("hip_l", "knee_l", "hip_r", "knee_r")          # indego_sim's exo joint order
EMG_MUSCLES = ("RF", "SOL", "VAS", "TA", "TFL", "BF", "GAS", "GLU")   # sensors 1-8 right leg, 9-16 left leg
# exo-log blocks [s]: 20 s after each switch are left out (transition)
BLOCKS = {"passive": (20.0, 120.0), "ver": (140.0, 240.0), "passive_after": (260.0, None)}
# noexo recording: low, mid, high speed back to back, ~120 s each
NOEXO_BLOCKS = {1: (15.0, 115.0), 2: (135.0, 235.0), 3: (255.0, 355.0)}
N_PHASE = 101


# ---------------------------------------------------------------------------------------------------------- files

def find_file(root: str, subject: int, kind: str, speed: Optional[int] = None) -> Optional[str]:
    """Path of one file, e.g. kind "exo"/"marker"/"emg"/"grf"/"imu" with speed 1-3, or speed None for noexo."""
    tag = f"sub_{subject}_noexo_{kind}" if speed is None else f"sub_{subject}_{kind}_v{speed}"
    if subject == 9 and kind == "exo" and speed is not None:
        tag = f"sub9_exo_v{speed + 1}"                       # see the module docstring
    ext = "mat" if kind == "exo" else "csv"
    hits = glob.glob(os.path.join(root, "**", f"{tag}.{ext}"), recursive=True)
    return hits[0] if hits else None


def load_exo(path: str) -> Dict[str, np.ndarray]:
    """The exo log in indego_sim's order (hip_l, knee_l, hip_r, knee_r): q [deg], qd [deg/s], tau_cmd [N*m],
    energy; thigh IMUs with the sagittal gyro axis signed so that + is thigh flexion on both sides."""
    m = loadmat(path, squeeze_me=True)
    keys = {"hip_l": "HL", "knee_l": "KL", "hip_r": "HR", "knee_r": "KR"}
    ce = m["ctrl_en"] if "ctrl_en" in m else np.asarray(m["switches"])[:, 1]     # subject 9: raw workspace
    out = dict(t=np.asarray(m["time"], float), ctrl_en=np.asarray(ce, int))
    for pre, name in (("Q", "q"), ("W", "qd"), ("F", "tau_cmd"), ("E", "energy")):
        out[name] = np.column_stack([np.asarray(m[pre + keys[j]], float) for j in JOINTS])
    if "GYL" in m:
        out["gyr_thigh_l"] = np.asarray(m["GYL"], float) * np.array([1, 1, -1])
        out["gyr_thigh_r"] = np.asarray(m["GYR"], float)
        out["acc_thigh_l"] = np.asarray(m["ACL"], float)
        out["acc_thigh_r"] = np.asarray(m["ACR"], float)
    out["fs"] = 1.0 / float(np.median(np.diff(out["t"])))
    return out


def read_vicon_csv(path: str):
    """A Vicon Nexus export (markers, GRF, EMG or IMU): (rate [Hz], group names, column names, data) with empty
    fields (marker gaps) as NaN. The first two columns are Frame and Sub Frame."""
    with open(path, encoding="utf-8-sig") as f:
        f.readline()
        rate = float(f.readline().split(",")[0])
        groups = f.readline().rstrip("\r\n").split(",")
        cols = f.readline().rstrip("\r\n").split(",")
        f.readline()                                           # units
        text = f.read()
    end = re.search(r"^(?![0-9])", text, flags=re.M)          # a second section (e.g. markers after the devices)
    if end is not None and end.start() < len(text):
        text = text[:end.start()]
    n = len(cols)
    while n > 2 and cols[n - 1] == "":
        n -= 1
    if re.search(r",,|,$|^,", text, flags=re.M):              # empty fields: marker gaps, trailing columns
        text = re.sub(r"(?<=,)(?=,|$)", "nan", text, flags=re.M)
        text = re.sub(r"^(?=,)", "nan", text, flags=re.M)
    data = np.loadtxt(text.splitlines(), delimiter=",", usecols=range(n), ndmin=2)
    data = data[~np.isnan(data[:, :2]).any(axis=1)]             # rows without a frame / sub-frame number
    return rate, groups[:n], cols[:n], data


def vicon_time(data: np.ndarray, rate: float, frame_rate: float = 100.0) -> np.ndarray:
    """Seconds from the first frame (Frame counts at the mocap rate, Sub Frame at the device rate)."""
    return (data[:, 0] - data[0, 0]) / frame_rate + data[:, 1] / rate


def markers(path: str) -> Dict[str, np.ndarray]:
    """Marker trajectories [mm] by name (lab frame: x right, y forward, z up), plus "t" [s]."""
    rate, groups, cols, data = read_vicon_csv(path)
    out = {"t": vicon_time(data, rate, rate)}
    for k, g in enumerate(groups):
        if g:
            name = g.split(":")[-1]
            out[name] = data[:, k:k + 3]
    for k, v in out.items():
        if k != "t":
            out[k] = _fill_gaps(v)
    return out


def _fill_gaps(x: np.ndarray, max_gap: int = 20) -> np.ndarray:
    """Linear interpolation over short marker gaps (<= max_gap frames); longer gaps stay NaN."""
    x = x.copy()
    for c in range(x.shape[1]):
        v = x[:, c]
        bad = np.isnan(v)
        if not bad.any() or bad.all():
            continue
        idx = np.arange(v.size)
        filled = np.interp(idx, idx[~bad], v[~bad])
        # keep long gaps as NaN
        edges = np.diff(np.r_[0, bad.astype(int), 0])
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if e - s <= max_gap:
                v[s:e] = filled[s:e]
        x[:, c] = v
    return x


# ------------------------------------------------------------------------------------------- kinematics, events

def lowpass(x, fs, fc, order=2):
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, x, axis=0)


def marker_angles(mk: Dict[str, np.ndarray], side: str = "R") -> Dict[str, np.ndarray]:
    """Sagittal hip and knee flexion [deg] from the 16-marker lower-body set: hip joint centre by Harrington et al.
    (2007) from the pelvis markers, thigh = HJC -> KNE, shank = KNE -> ANK. Knee = 3-D angle between thigh and shank
    (0 = straight); hip = thigh direction in the pelvis frame's sagittal plane (0 = vertical, + = flexion)."""
    lasi, rasi, lpsi, rpsi = mk["LASI"], mk["RASI"], mk["LPSI"], mk["RPSI"]
    mid_asis, mid_psis = (lasi + rasi) / 2, (lpsi + rpsi) / 2
    z = _unit(rasi - lasi)                                     # right
    x = mid_asis - mid_psis
    x = _unit(x - np.sum(x * z, 1, keepdims=True) * z)         # anterior
    y = np.cross(z, x)                                         # superior
    pw = np.linalg.norm(rasi - lasi, axis=1, keepdims=True)
    pd = np.linalg.norm(mid_asis - mid_psis, axis=1, keepdims=True)
    sgn = 1.0 if side == "R" else -1.0
    hjc = mid_asis + x * (-0.24 * pd - 9.9) + y * (-0.30 * pw - 10.9) + z * sgn * (0.33 * pw + 7.3)
    kne, ank = mk[side + "KNE"], mk[side + "ANK"]
    th, sh = _unit(kne - hjc), _unit(ank - kne)
    knee = np.degrees(np.arccos(np.clip(np.sum(th * sh, 1), -1, 1)))
    hip = np.degrees(np.arctan2(np.sum(th * x, 1), -np.sum(th * y, 1)))
    return {"hip": hip, "knee": knee}


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def hip_peaks(hip: np.ndarray, fs: float) -> np.ndarray:
    """Cycle starts = hip flexion maxima (late swing, just before heel strike) - the event both the real encoders
    and the simulated ones give the same way."""
    h = lowpass(hip, fs, 6.0)
    pk, _ = find_peaks(h, distance=int(0.6 * fs), prominence=12.0)
    return pk


def heel_strikes(load: np.ndarray, fs: float, on: float = 50.0, min_swing: float = 0.15,
                 min_stance: float = 0.15) -> np.ndarray:
    """Indices where the vertical load [N] on one foot / belt rises above `on` after >= min_swing s unloaded and
    stays loaded for most of the next min_stance s (a foot brushing the ground in swing is not a heel strike)."""
    c = lowpass(load, fs, 20.0) > on
    rise = np.flatnonzero(c[1:] & ~c[:-1]) + 1
    keep = []
    off = np.flatnonzero(~c[1:] & c[:-1]) + 1
    w = max(1, int(round(min_stance * fs)))
    for r in rise:
        prev_off = off[off < r]
        if prev_off.size and (r - prev_off[-1]) / fs >= min_swing and c[r:r + w].mean() >= 0.8:
            keep.append(r)
    keep = np.array(keep, int)
    if len(keep) >= 4:                                     # drop re-contacts well inside a stride
        med, out = np.percentile(np.diff(keep), 75), [keep[0]]
        for r in keep[1:]:
            if r - out[-1] >= 0.6 * med:
                out.append(r)
        keep = np.array(out, int)
    return keep


def good_strides(starts: np.ndarray, fs: float, tol: float = 0.25) -> np.ndarray:
    """(n, 2) start/end indices of strides within +-tol of the median duration."""
    if starts.size < 3:
        return np.zeros((0, 2), int)
    pairs = np.column_stack([starts[:-1], starts[1:]])
    dur = np.diff(pairs, axis=1)[:, 0]
    med = np.median(dur)
    return pairs[np.abs(dur / med - 1) < tol]


def normalise(x: np.ndarray, strides: np.ndarray, n: int = N_PHASE) -> np.ndarray:
    """(strides, n[, channels]) time-normalised cycles."""
    ph = np.linspace(0, 1, n)
    out = []
    for s, e in strides:
        seg = x[s:e + 1]
        u = np.linspace(0, 1, len(seg))
        out.append(np.stack([np.interp(ph, u, seg[:, c]) for c in range(seg.shape[1])], -1) if seg.ndim > 1
                   else np.interp(ph, u, seg))
    return np.array(out)


def joint_features(hip_c: np.ndarray, knee_c: np.ndarray) -> Dict[str, float]:
    """Per-stride means on hip-peak cycles: ROM, peak hip flexion (swing), peak hip extension, knee stance extension
    (minimum flexion, 15-60 % of the cycle) and peak knee swing flexion (50-95 %)."""
    return {
        "hip_rom": float(np.mean(np.ptp(hip_c, 1))),
        "hip_max": float(np.mean(hip_c.max(1))),
        "hip_min": float(np.mean(hip_c.min(1))),
        "knee_rom": float(np.mean(np.ptp(knee_c, 1))),
        "knee_stance_min": float(np.mean(knee_c[:, 15:61].min(1))),
        "knee_swing_max": float(np.mean(knee_c[:, 50:96].max(1))),
    }


def pelvis_ranges(t: np.ndarray, R: np.ndarray, pos: np.ndarray, fwd: int = 0, up: int = 1,
                  t_skip: float = 3.0) -> Dict[str, float]:
    """Ranges (2nd-98th percentile, deg) of the pelvis heading relative to the walking direction (yaw), its pitch and
    its roll, after the first t_skip s. R: (n, 3, 3) pelvis rotation matrices (columns = body axes in the world,
    z up); pos: (n, 3) pelvis positions; the walking direction is the pelvis path over ~1.4 s. MyoLeg's pelvis has
    x forward and y up. People at 0.5-1 m/s: yaw ~10-15, pitch ~3-5, roll ~5-8 deg (more in an exo)."""
    from scipy.ndimage import uniform_filter1d
    fs = 1.0 / np.median(np.diff(t))
    v = uniform_filter1d(np.gradient(pos[:, :2], axis=0), max(1, int(round(1.4 * fs))), axis=0)
    f = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    lat = np.stack([-f[:, 1], f[:, 0]], 1)
    yaw = np.angle(np.exp(1j * (np.arctan2(R[:, 1, fwd], R[:, 0, fwd]) - np.arctan2(f[:, 1], f[:, 0]))))
    u = R[:, :, up]
    w = t > t[0] + t_skip
    rng = lambda x: float(np.ptp(np.percentile(np.degrees(x[w]), [2, 98])))   # noqa: E731
    return {"pelvis_yaw_range": rng(yaw),
            "pelvis_pitch_range": rng(np.arcsin(np.clip((u[:, :2] * f).sum(1), -1, 1))),
            "pelvis_roll_range": rng(np.arcsin(np.clip((u[:, :2] * lat).sum(1), -1, 1)))}


def sagittal_gyro_gain(gyro: np.ndarray, hip_qd: np.ndarray, fs: float, max_lag: float = 0.1):
    """Thigh gyro (sagittal axis, deg/s) vs encoder hip velocity: best correlation, lag [s] (+: the gyro leads),
    regression gain. A gain < 1 means the hip unit pitches against the thigh swing."""
    lags = np.arange(-int(max_lag * fs), int(max_lag * fs) + 1)
    n = len(gyro)
    cut = slice(len(lags), n - len(lags))
    xc = [np.corrcoef(gyro[cut], np.roll(hip_qd, k)[cut])[0, 1] for k in lags]
    k = lags[int(np.argmax(xc))]
    w = np.roll(hip_qd, k)[cut]
    return float(max(xc)), float(-k / fs), float(np.polyfit(w, gyro[cut], 1)[0])


def sync_offset(t_exo: np.ndarray, knee_exo: np.ndarray, t_vic: np.ndarray, knee_vic: np.ndarray,
                fs: float = 100.0, min_overlap: float = 60.0):
    """Offset [s] with t_exo = t_vicon + offset, from the cross-correlation of the encoder and the marker knee angle.
    Returns (offset, r at the best lag, best r at least 0.4 s away from it) - a clear margin means a sure match."""
    ge = np.arange(t_exo[0], t_exo[-1], 1 / fs)
    gv = np.arange(t_vic[0], t_vic[-1], 1 / fs)
    ke = np.interp(ge, t_exo, knee_exo)
    kv = np.interp(gv, t_vic[~np.isnan(knee_vic)], knee_vic[~np.isnan(knee_vic)])
    a, b = (kv - kv.mean()) / kv.std(), (ke - ke.mean()) / ke.std()
    c = correlate(a, b, mode="full", method="fft")
    lags = correlation_lags(len(a), len(b), mode="full")
    ov = np.minimum(len(a), lags + len(b)) - np.maximum(0, lags)
    ok = ov >= min_overlap * fs
    score = np.where(ok, c / np.maximum(ov, 1), -np.inf)

    def pearson(k):
        i0, i1 = max(0, k), min(len(a), k + len(b))
        return float(np.corrcoef(a[i0:i1], b[i0 - k:i1 - k])[0, 1])

    k = int(lags[np.argmax(score)])
    r = pearson(k)
    far = np.abs(lags - k) > int(0.4 * fs)
    k2 = int(lags[np.argmax(np.where(far, score, -np.inf))])
    return float(ge[0] - gv[0] - k / fs), r, pearson(k2)


# ------------------------------------------------------------------------------------------------------- EMG

def emg_envelope(emg: np.ndarray, fs: float) -> np.ndarray:
    """The paper's processing: band-pass 25-500 Hz, full-wave rectification, 100-sample moving average."""
    b, a = butter(4, [25 / (fs / 2), min(500 / (fs / 2), 0.99)], btype="band")
    r = np.abs(filtfilt(b, a, emg, axis=0))
    k = np.ones(100) / 100
    return np.stack([np.convolve(r[:, c], k, mode="same") for c in range(r.shape[1])], -1)


def on_off(curve: np.ndarray, frac: float = 0.3) -> np.ndarray:
    """Boolean 'active' mask of a mean cycle curve: above min + frac * (max - min)."""
    lo, hi = curve.min(), curve.max()
    return curve > lo + frac * (hi - lo)


# ------------------------------------------------------------------------------------- recording a simulated trial

SIM_MUSCLES = {"RF": ["recfem"], "SOL": ["soleus"], "VAS": ["vasmed"], "TA": ["tibant"], "TFL": ["tfl"],
               "BF": ["bflh"], "GAS": ["gasmed"], "GLU": ["glmax1", "glmax2", "glmax3"]}   # MyoLeg for each EMG site


class TrialRecorder:
    """Records an indego_sim walk at the control rate in the trial format the Waterloo comparison reads: q, qd
    (n, 4) deg, deg/s from the exo encoders (hip_l, knee_l, hip_r, knee_r); thigh gyro (deg/s) and accelerometer (g);
    vertical foot loads [N] (l, r); muscle excitations (d.ctrl) at the EMG sites in exc_order; the person's own hip and
    knee angles [deg]; COM x [m]; pelvis pitch [deg]. Call it with each observation; arrays() returns the trial."""

    def __init__(self, sim):
        self.sim = sim
        m, hc = sim.m, sim.human
        names = [m.actuator(i).name for i in hc.mm.ids]
        self.order = [f"{k}_{s}" for s in "rl" for k in EMG_MUSCLES]
        self.mus = {f"{k}_{s}": [hc.mm.ids[names.index(f"{p}_{s}")] for p in SIM_MUSCLES[k]] for k in EMG_MUSCLES
                    for s in "rl"}
        self.dof = [hc.qadr[hc.dof_names.index(k)] for k in
                    ("hip_flexion_l", "knee_angle_l", "hip_flexion_r", "knee_angle_r")]
        self.pelvis = m.body("pelvis").id
        self.rec = {k: [] for k in ("q", "qd", "gyr_l", "gyr_r", "acc_l", "acc_r", "load", "exc", "q_human",
                                    "com_x", "pelvis_pitch")}

    def __call__(self, obs):
        sim, d, r = self.sim, self.sim.d, self.rec
        r["q"].append(np.degrees(obs["q"]))
        r["qd"].append(np.degrees(obs["qd"]))
        r["gyr_l"].append(np.degrees(obs["gyr_thigh_l"]))
        r["gyr_r"].append(np.degrees(obs["gyr_thigh_r"]))
        r["acc_l"].append(obs["acc_thigh_l"] / 9.81)
        r["acc_r"].append(obs["acc_thigh_r"] / 9.81)
        r["load"].append([max(float(sim._grf[s][2]), 0.0) for s in "lr"])
        r["exc"].append([np.mean(d.ctrl[self.mus[k]]) for k in self.order])
        r["q_human"].append(np.degrees(d.qpos[self.dof]))
        r["com_x"].append(float(d.subtree_com[sim.root][0]))
        R = d.xmat[self.pelvis].reshape(3, 3)
        r["pelvis_pitch"].append(float(np.degrees(np.arctan2(R[2, 0], R[0, 0]))))

    def arrays(self, **extra):
        out = {k: np.asarray(v, float) for k, v in self.rec.items()}
        out.update(fs=1.0 / self.sim.ctrl_dt, fell=bool(self.sim.fell), exc_order=np.array(self.order), **extra)
        return out
