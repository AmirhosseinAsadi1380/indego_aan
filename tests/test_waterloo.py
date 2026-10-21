import numpy as np

from indego_aan import waterloo as W


def _gait(T=120.0, fs=100.0, seed=0):
    """Synthetic hip and knee angles [deg], stride times 1.2 s +- 3 % (stride to stride)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, 1 / fs)
    ends = np.cumsum(rng.normal(1.2, 0.036, int(T / 1.0)))
    ph = 2 * np.pi * np.interp(t, np.r_[0, ends], np.arange(ends.size + 1))
    hip = 10 + 22 * np.cos(ph)
    amp = np.repeat(rng.normal(1, 0.08, int(T) + 2), int(fs))[:t.size]   # shape varies second to second
    knee = 30 - 25 * amp * np.cos(2 * ph + 0.6) + 5 * np.sin(ph) + rng.normal(0, 1.0, t.size)
    return t, hip, knee


def test_vicon_csv_with_gaps_and_trailing_columns(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("Trajectories\n100\n,,A,,,B,,,\nFrame,Sub Frame,X,Y,Z,X,Y,Z,,\n,,mm,mm,mm,mm,mm,mm,,\n"
                 "1,0,1,2,3,4,5,6,,\n2,0,1.5,,3.5,4,5,6,,\n3,0,2,3,4,4,5,6,,\n", encoding="utf-8")
    rate, groups, cols, data = W.read_vicon_csv(str(p))
    assert rate == 100 and data.shape == (3, 8) and np.isnan(data[1, 3])
    mk = W.markers(str(p))
    assert np.allclose(mk["A"][1], [1.5, 2.5, 3.5]) and np.allclose(mk["t"], [0, 0.01, 0.02])


def test_strides_and_features_on_a_synthetic_gait():
    t, hip, knee = _gait()
    st = W.good_strides(W.hip_peaks(hip, 100.0), 100.0)
    assert len(st) > 90 and abs(np.median(np.diff(st, axis=1)) / 100.0 - 1.2) < 0.05
    f = W.joint_features(W.normalise(hip, st), W.normalise(knee, st))
    assert abs(f["hip_rom"] - 44) < 1.5 and abs(f["hip_max"] - 32) < 1.0


def test_sync_offset_recovers_the_clock_shift():
    t, hip, knee = _gait(T=200.0)
    sel = (t > 30) & (t < 180)
    t_exo = t[sel] - 30.0                                   # the exo log starts 30 s into the mocap recording
    o, r, r2 = W.sync_offset(t_exo, knee[sel], t, knee)
    assert abs(o - (-30.0)) < 0.02 and r > 0.99 and r - r2 > 0.05


def test_heel_strikes_from_a_belt_load():
    fs = 1000.0
    t = np.arange(0, 10, 1 / fs)
    load = np.where((t % 1.2) < 0.75, 700.0, 0.0)           # stance 0.75 s of every 1.2 s
    hs = W.heel_strikes(load, fs)
    assert len(hs) == 8 and np.allclose(np.diff(hs) / fs, 1.2, atol=0.01)


def test_a_foot_brushing_the_ground_in_swing_is_not_a_heel_strike():
    from indego_aan import metrics
    fs, stride = 200.0, 1.4
    t = np.arange(0, 20, 1 / fs)
    ph = (t % stride) / stride
    load = np.where(ph < 0.6, 700.0, 0.0)                    # stance 60 %
    load[(ph > 0.80) & (ph < 0.83)] = 120.0                   # a brief toe brush in mid-swing
    rec = {"load": np.column_stack([load, load]), "body_weight": 750.0, "dt": 1 / fs, "t": t}
    hs = metrics.heel_strikes(rec, 1)
    assert len(hs) == 14 and np.allclose(np.diff(hs) / fs, stride, atol=0.01)
    hs2 = W.heel_strikes(load, fs)
    assert np.allclose(np.diff(hs2) / fs, stride, atol=0.02)


def test_pelvis_ranges_of_a_twisting_pelvis():
    from scipy.spatial.transform import Rotation as R
    fs = 100.0
    t = np.arange(0, 20, 1 / fs)
    ph = 2 * np.pi * 0.7 * t                                  # stride frequency
    yaw, pitch, roll = np.radians(20) * np.sin(ph), np.radians(2) * np.sin(2 * ph), np.radians(3) * np.cos(ph)
    rest = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])     # MyoLeg pelvis (x forward, y up, z right), walking along x
    rot = R.from_euler("zyx", np.column_stack([yaw, pitch, roll])).as_matrix() @ rest
    pos = np.column_stack([1.0 * t, 0.02 * np.sin(ph), np.full_like(t, 0.9)])
    r = W.pelvis_ranges(t, rot, pos)
    assert abs(r["pelvis_yaw_range"] - 40) < 1.5 and abs(r["pelvis_pitch_range"] - 4) < 1.0
    assert abs(r["pelvis_roll_range"] - 6) < 1.0
