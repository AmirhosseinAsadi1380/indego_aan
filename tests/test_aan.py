import numpy as np
import pytest

from indego_aan.deficits import Deficits, DriveProcess, Fatigue3CCr
from indego_aan.metrics import GaitBaseline, evaluate, heel_strikes, gait_deviation_scores

MUSCLES = ["vasint_r", "soleus_r", "tibant_r", "glmax1_r"]


def test_fatigue_accumulates_under_load_and_recovers_at_rest():
    f = Fatigue3CCr(MUSCLES)
    full = np.ones(4)
    for _ in range(60000):                                   # 60 s at full activation, 1 ms steps
        u = f.limit(full, np.minimum(full, 1 - f.MF), 1e-3)
    mf_loaded = f.MF.copy()
    assert np.all(mf_loaded > 0.2) and np.all(mf_loaded < 0.6)   # F ~ 0.007-0.01 /s: ~30-45 % after 60 s at MVC
    assert np.allclose(u, 1 - f.MF)                           # cannot be excited above the unfatigued fraction
    for _ in range(60000):                                   # 60 s of rest
        f.limit(np.zeros(4), np.zeros(4), 1e-3)
    assert np.all(f.MF < mf_loaded)


def test_fatigue_groups_and_rate_scale():
    f1, f10 = Fatigue3CCr(MUSCLES), Fatigue3CCr(MUSCLES, rate_scale=10.0)
    assert f1.group == ["knee_extensors", "ankle_plantarflexors", "ankle_dorsiflexors", "hip_extensors"]
    assert np.allclose(f10.F, 10 * f1.F) and np.allclose(f10.R, 10 * f1.R)


def test_drive_process_bounds_schedule_and_lapses():
    d = DriveProcess(schedule=[(0, 1.0), (10, 1.0), (15, 0.5), (1e9, 0.5)])
    assert d.value(5, 0.01) == 1.0 and d.value(12.5, 0.01) == pytest.approx(0.75) and d.value(20, 0.01) == 0.5
    d = DriveProcess(mean=0.8, drift_sd=0.3, lapse_rate=2.0, seed=1)
    v = np.array([d.value(k * 0.01, 0.01) for k in range(3000)])
    assert v.min() >= 0.3 and v.max() <= 1.0 and v.std() > 0.05
    d = DriveProcess(schedule=[(0, 0.6), (1e9, 0.6)], drift_sd=0.1, seed=2)       # drift on top of a schedule
    v = np.array([d.value(k * 0.01, 0.01) for k in range(3000)])
    assert abs(v.mean() - 0.6) < 0.1 and v.std() > 0.02


def _synthetic_rec(n_steps=40, step_time=0.5, dt=0.005, work_h=1.0, work_e=0.25):
    t = np.arange(int(n_steps * step_time / dt)) * dt
    phase = (t / step_time) % 2.0
    load = np.stack([(phase >= 1.0), (phase < 1.0)], axis=1) * 600.0     # alternating single support
    q = np.stack([np.sin(2 * np.pi * t / (2 * step_time) + k) for k in range(6)], axis=1)
    qd = np.ones((len(t), 6))
    return {"t": t, "dt": dt, "load": load, "q": q, "qd": qd, "body_weight": 700.0, "fell": False,
            "chan": [f"{j}_{s}" for s in "lr" for j in ("hip_flexion", "knee_angle", "ankle_angle")],
            "tau_h": np.full((len(t), 6), work_h), "tau_exo": np.full((len(t), 4), work_e), "qd_exo": np.ones((len(t), 4)),
            "com_vx": np.full(len(t), 0.6), "drive": np.ones(len(t)), "fatigue": np.zeros(len(t))}


def test_metrics_on_a_synthetic_gait():
    rec = _synthetic_rec()
    hs = heel_strikes(rec, 1)
    assert len(hs) >= 18 and np.allclose(np.diff(rec["t"][hs]), 1.0)
    base = GaitBaseline.from_records([rec])
    assert base.step_time == pytest.approx(0.5) and base.speed == pytest.approx(0.6)
    e = evaluate(rec, base)
    assert e["gait_deviation"] < 0.1                         # the baseline itself
    # hip + knee: 4 human channels at 1 W each, 4 exo joints at 0.25 W each -> share 0.8
    assert e["human_share"] == pytest.approx(0.8)


def test_gait_deviation_passes_on_the_actuated_joints():
    chan = [f"{j}_{s}" for s in "lr" for j in ("hip_flexion", "knee_angle", "ankle_angle")]
    g = gait_deviation_scores(chan, [0.5, 0.8, 4.0, 0.6, 0.9, 3.0])        # ankles off, hips and knees in band
    assert g["gait_deviation"] == pytest.approx(0.9) and g["gait_deviation_ankle"] == pytest.approx(4.0)
    assert g["gait_deviation_all"] == pytest.approx(4.0)


@pytest.mark.slow
def test_deficits_attach_to_the_walking_person():
    from indego_aan.scenario import make_walking_sim
    sim, obs = make_walking_sim(seed=0)
    defi = Deficits(sim, drive=DriveProcess(schedule=[(0, 0.5), (1e9, 0.5)]), mode="command")
    for _ in range(100):                                     # 0.5 s
        obs = sim.step(np.zeros(4), record=False)
    hc = sim.human
    assert defi.drive_now == 0.5
    u = sim.d.ctrl[hc.mm.ids]
    assert np.all(u <= np.clip(0.5 * hc.a_vol + hc.e_reflex, 0, 1) + 1e-9)
    defi.detach()
    assert hc.step == hc.__class__.step.__get__(hc)


@pytest.mark.slow
def test_teacher_env_runs_and_rewards_fresh_walking():
    from indego_aan.env import AANEnv
    env = AANEnv(deficit={"kind": "fresh"}, episode_s=3.0)
    obs, _ = env.reset(options={"sim_seed": 0})
    assert obs.shape == env.observation_space.shape and np.all(np.isfinite(obs))
    rs = []
    while True:
        obs, r, term, trunc, info = env.step(np.zeros(4))
        rs.append(r)
        if term or trunc:
            break
    assert not term and np.all(np.isfinite(rs))
    assert info["r_stab"] == pytest.approx(1.0) and info["share"] > 0.95       # fresh, unassisted, stable


@pytest.mark.slow
def test_teacher_env_resets_do_not_compound_the_muscle_scaling():
    from indego_aan.env import AANEnv
    from indego_aan.scenario import make_walking_sim
    sim, _ = make_walking_sim(seed=0)
    F0 = sim.human.mm.F0.copy()                        # muscle strengths after the person's scaling, built fresh
    env = AANEnv(deficit={"kind": "fresh"}, episode_s=0.1)
    for _ in range(3):
        env.reset(options={"sim_seed": 0})
        assert np.allclose(env.sim.human.mm.F0, F0)


@pytest.mark.slow
def test_teacher_env_with_the_personal_aan_base():
    from indego_aan.env import env_from_config
    env = env_from_config({"base": "paan"}, deficit={"kind": "fresh"}, episode_s=2.0)
    obs, _ = env.reset(options={"sim_seed": 0})
    assert obs.shape == env.observation_space.shape == (80,)
    while True:
        obs, r, term, trunc, info = env.step(np.zeros(4))
        if term or trunc:
            break
    assert not term and np.allclose(info["tau"], np.clip(info["tau_base"], -40, 40))   # zero correction = the AAN
