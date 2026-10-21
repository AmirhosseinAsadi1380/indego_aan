"""Results showcase: animated scenarios (GIF + MP4) and summary figures from the saved results.

    python scripts/showcase.py [out_dir=docs/media] [aan] [estimate] [kinesis] [figures]   (default: aan estimate)

The GIFs (README-sized: trimmed, scaled, plain floor) and the figures go to out_dir; full-resolution MP4s are written
next to them.

* aan       - the simulated person (T12 AIS D, own walking, sagittal) whose volitional drive drops from 100 to 40 %
              between 10 and 15 s, side by side: zero torque, transparent mode, the personal AAN (seed 0, the
              baselines' setup).
* estimate  - the personal AAN in the same scenario with live plots: true capacity vs the linear estimate from the
              controller's own signals (results/deficit_estimate.json), the gains and the exo torque.
* kinesis   - the learned human (KINESIS, fine-tuned with the exo) in the Indego, seen from above: the exo's frontal-plane hip springs
              relaxed vs on (600 / 150 N*m/rad); arrows show the pelvis heading (red) and the walking direction
              (green). Needs the KINESIS checkout (KINESIS_DIR) and the fine-tuned policy folder (KINESIS_MODEL).
* figures   - PNG summaries of results/*.json.

The renders show indego_sim's Indego model: keep them out of public places if the exo mesh is the licensed one.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FPS = 10                                   # rendered frames per simulated second (played in real time)


# --------------------------------------------------------------------------------------------------- drawing

def _font(size, bold=False):
    from PIL import ImageFont
    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        for d in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", ""):
            try:
                return ImageFont.truetype(os.path.join(d, name) if d else name, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _bar(draw, x, y, w, h, frac, color, back=(225, 225, 225)):
    draw.rectangle([x, y, x + w, y + h], fill=back)
    draw.rectangle([x, y, x + int(w * min(max(frac, 0.0), 1.0)), y + h], fill=color)


def _title(width, text, sub=""):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (width, 52 if sub else 32), (30, 40, 60))
    dr = ImageDraw.Draw(im)
    dr.text((10, 5), text, fill=(255, 255, 255), font=_font(18, True))
    if sub:
        dr.text((10, 29), sub, fill=(200, 210, 230), font=_font(14))
    return im


def _save(frames, path_base, fps=FPS, gif_from=0.0, scale=0.75, colors=128):
    """GIF (from gif_from seconds, scaled, few colours: small enough for a README) and the full MP4."""
    import imageio
    from PIL import Image
    h, w = frames[0].shape[:2]
    size = (int(w * scale), int(h * scale))
    pal = [Image.fromarray(f).resize(size, Image.LANCZOS).quantize(colors=colors, method=Image.Quantize.FASTOCTREE,
                                                                    dither=Image.Dither.NONE)
           for f in frames[int(round(gif_from * fps)):]]
    pal[0].save(path_base + ".gif", save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0,
                optimize=True)
    imageio.mimsave(path_base + ".mp4", [f[:f.shape[0] // 2 * 2, :f.shape[1] // 2 * 2] for f in frames], fps=fps,
                    quality=8, macro_block_size=2)
    print("->", path_base + ".gif", f"({os.path.getsize(path_base + '.gif') / 1e6:.1f} MB)", "and .mp4", flush=True)


class Camera:
    """Offscreen renderer with a camera that follows the person."""

    def __init__(self, sim, w, h, azimuth=90.0, distance=3.0, elevation=-8.0, lookat_z=0.8, plain_floor=True):
        import mujoco
        self.sim, self.mj = sim, mujoco
        m = sim.m
        if plain_floor:                                      # visual only: a flat floor compresses far better in GIFs
            for i in range(m.ngeom):
                if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                    m.geom_matid[i] = -1
                    m.geom_rgba[i] = (0.86, 0.87, 0.89, 1.0)
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
        m.vis.global_.offheight = max(m.vis.global_.offheight, h)
        self.r = mujoco.Renderer(m, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.azimuth, self.cam.distance, self.cam.elevation = azimuth, distance, elevation
        self.cam.lookat[:] = sim.d.subtree_com[sim.root]
        self.cam.lookat[2] = lookat_z
        self.arrows = []

    def render(self, follow=0.15):
        com = self.sim.d.subtree_com[self.sim.root]
        self.cam.lookat[:2] += follow * (com[:2] - self.cam.lookat[:2])
        self.r.update_scene(self.sim.d, self.cam)
        sc = self.r.scene
        for a, b, rgba in self.arrows:
            if sc.ngeom < sc.maxgeom:
                g = sc.geoms[sc.ngeom]
                self.mj.mjv_initGeom(g, self.mj.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                                     np.asarray(rgba, np.float32))
                self.mj.mjv_connector(g, self.mj.mjtGeom.mjGEOM_ARROW, 0.03, a, b)
                sc.ngeom += 1
        return self.r.render().copy()


# ------------------------------------------------------------------------------------- scenario 1: controllers

DRIVE = [(0.0, 1.0), (10.0, 1.0), (15.0, 0.4), (1e9, 0.4)]


def _aan_sims(names, seed=0):
    from baselines import controller
    from indego_aan.deficits import Deficits, DriveProcess
    from indego_aan.scenario import make_walking_sim
    out = []
    for name in names:
        sim, obs = make_walking_sim(seed=seed)
        defi = Deficits(sim, drive=DriveProcess(schedule=DRIVE))
        out.append(dict(name=name, sim=sim, obs=obs, defi=defi, ctrl=controller(name, sim), tau=np.zeros(4),
                        fell_at=None, frame=None))
    return out


def _strip(width, s):
    """Under a panel: the controller and whether the person is still walking."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (width, STRIP), (248, 248, 248))
    dr = ImageDraw.Draw(im)
    label = {"zero": "Motors off", "transparent": "Transparent mode", "paan": "Personal assist-as-needed"}[s["name"]]
    dr.text((8, 6), label, fill=(20, 20, 20), font=_font(15, True))
    fell = s["fell_at"] is not None
    dr.text((width - (60 if fell else 80), 6), "fell" if fell else "walking", fill=(200, 30, 30) if fell else
            (30, 140, 60), font=_font(15, True))
    return im


STRIP = 30


def scenario_aan(out, T=26.0, w=360, h=330):
    from PIL import Image
    sims = _aan_sims(["zero", "transparent", "paan"])
    cams = [Camera(s["sim"], w, h) for s in sims]
    dt = sims[0]["sim"].ctrl_dt
    every = int(round(1.0 / FPS / dt))
    frames = []
    for k in range(int(round(T / dt))):
        for s in sims:
            if s["fell_at"] is not None and s["sim"].d.time - s["sim"].t0 > s["fell_at"] + 1.0:
                continue                                       # frozen one second after the fall
            s["tau"] = np.asarray(s["ctrl"](s["obs"]), float)
            s["obs"] = s["sim"].step(s["tau"], record=False)
            if s["sim"].fell and s["fell_at"] is None:
                s["fell_at"] = s["sim"].d.time - s["sim"].t0
        if k % every:
            continue
        t = k * dt
        panels = []
        for s, c in zip(sims, cams):
            frozen = s["fell_at"] is not None and s["sim"].d.time - s["sim"].t0 > s["fell_at"] + 1.0
            if not frozen or s["frame"] is None:
                s["frame"] = c.render()
            col = Image.new("RGB", (w, h + STRIP))
            col.paste(Image.fromarray(s["frame"]), (0, 0))
            col.paste(_strip(w, s), (0, h))
            panels.append(col)
        top = _title(3 * w, f"Voluntary drive {100 * sims[0]['defi'].drive_now:3.0f} %      t = {t:4.1f} s")
        frame = Image.new("RGB", (3 * w, top.height + h + STRIP), (255, 255, 255))
        frame.paste(top, (0, 0))
        for i, p in enumerate(panels):
            frame.paste(p, (i * w, top.height))
        frames.append(np.asarray(frame))
    _save(frames, os.path.join(out, "01_drive_drop_three_controllers"), gif_from=7.0)


# ------------------------------------------------------------------------------- scenario 2: capacity estimate

def scenario_estimate(out, T=30.0, w=440, h=380):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    s = _aan_sims(["paan"])[0]
    sim, c = s["sim"], s["ctrl"]
    cam = Camera(sim, w, h)
    coef = np.array(json.load(open(os.path.join(RES, "deficit_estimate.json")))["coef_all"])   # [1, k_hip, k_knee, |tau|, stride]
    dt = sim.ctrl_dt
    every = int(round(1.0 / FPS / dt))
    tau_ema, ph_prev, last_wrap, stride = 0.0, None, None, np.nan
    hist = {k: [] for k in ("t", "cap", "est", "tau")}
    fig, axs = plt.subplots(2, 1, figsize=(5.4, h / 100), dpi=100, sharex=True)
    frames = []
    for k in range(int(round(T / dt))):
        s["tau"] = np.asarray(c(s["obs"]), float)
        s["obs"] = sim.step(s["tau"], record=False)
        t = sim.d.time - sim.t0
        tau_ema += 0.01 * (float(np.mean(np.abs(s["tau"]))) - tau_ema)          # ~0.5 s, as deficit_estimate.py
        if c.phase is not None and ph_prev is not None and c.phase < ph_prev - 0.5:
            if last_wrap is not None:
                stride = t - last_wrap
            last_wrap = t
        ph_prev = c.phase
        if sim.fell:
            break
        if k % every:
            continue
        x = np.array([1.0, np.mean(c.k[[0, 2]]), np.mean(c.k[[1, 3]]), tau_ema, stride])
        est = float(coef @ x) if np.isfinite(stride) and t > 3 else np.nan
        for key, v in (("t", t), ("cap", s["defi"].drive_now), ("est", est), ("tau", tau_ema)):
            hist[key].append(v)
        for ax in axs:
            ax.cla()
            ax.set_xlim(0, T)
            ax.grid(alpha=0.3)
        H = {kk: np.array(v) for kk, v in hist.items()}
        axs[0].plot(H["t"], 100 * H["cap"], color="k", lw=2, label="capacity")
        axs[0].plot(H["t"], 100 * np.clip(H["est"], 0, 1.2), color="tab:red", lw=1.8, label="estimated by the exo")
        axs[0].set_ylim(0, 115)
        axs[0].set_ylabel("capacity [%]")
        axs[0].legend(loc="lower left", fontsize=9, frameon=False)
        axs[1].plot(H["t"], H["tau"], color="tab:orange", lw=1.8)
        axs[1].set_ylabel("exo torque [N*m]")
        axs[1].set_xlabel("time [s]")
        axs[1].set_ylim(0, max(4.0, 1.15 * H["tau"].max()))
        fig.tight_layout()
        fig.canvas.draw()
        plot = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        top = _title(w + plot.width, f"Personal assist-as-needed      t = {t:4.1f} s")
        frame = Image.new("RGB", (w + plot.width, top.height + max(h, plot.height)), (255, 255, 255))
        frame.paste(top, (0, 0))
        frame.paste(Image.fromarray(cam.render()), (0, top.height))
        frame.paste(plot, (w, top.height))
        frames.append(np.asarray(frame))
    plt.close(fig)
    _save(frames, os.path.join(out, "02_assist_as_needed_and_capacity_estimate"), gif_from=6.0)


# ------------------------------------------------------------------------------------ scenario 3: learned human

def _kinesis_run(exo_set, port, T, w, h, label):
    """Run the KINESIS person in the Indego for T s and return the rendered frames (pelvis-heading arrows drawn)."""
    import mujoco
    from PIL import Image, ImageDraw
    from indego_aan.kinesis_human import KinesisHuman
    from indego_sim.config import ExoParams, SimConfig, SupportParams
    from indego_sim.human.sci import get_preset
    from indego_sim.sim import HumanExoSim
    from kinesis_walk import start_server
    kin = os.environ.get("KINESIS_DIR", os.path.expanduser("~/kinesis"))
    model = os.environ.get("KINESIS_MODEL")
    motion, pose = "data/long_walks/long_walk_0.8.pkl", "data/long_walks/initial_pose_long_walk_0.8.pkl"
    server = start_server(kin, motion, pose, port, model)
    frames, info = [], {"fell_at": None}
    try:
        p = ExoParams()
        for item in filter(None, exo_set.split(",")):
            k, v = item.split("=")
            setattr(p, k, type(getattr(p, k))(v))
        sim = HumanExoSim(exo=p, subject=get_preset("healthy"), support=SupportParams(kind="none"),
                          sim=SimConfig(planar=False), with_exo=True, seed=0, log_muscles=False, log_rate_hz=20)
        m = sim.m
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        for j in ("lumbar_flexion", "lumbar_bending", "lumbar_rotation"):              # KINESIS's rigid trunk
            jid = m.joint(j).id
            m.jnt_stiffness[jid] = 3000.0
            m.qpos_spring[m.jnt_qposadr[jid]] = m.key("stand").qpos[m.jnt_qposadr[jid]]
            m.dof_damping[m.jnt_dofadr[jid]] = 100.0
        sim.reset(settle=1.0)
        person = KinesisHuman(sim, port=port)
        person.start()
        d, pel = sim.d, m.body("pelvis").id
        hr = [m.jnt_qposadr[m.joint(j).id] for j in ("hip_rotation_r", "hip_rotation_l")]
        cam = Camera(sim, w, h, azimuth=25.0, distance=2.4, elevation=-55.0, lookat_z=1.35)
        dt = sim.ctrl_dt
        every = int(round(1.0 / FPS / dt))
        path, yaws = [], []
        for k in range(int(round(T / dt))):
            sim.step(np.zeros(4), record=False)
            t = d.time - sim.t0
            path.append(d.xpos[pel][:2].copy())
            if sim.fell and info["fell_at"] is None:
                info["fell_at"] = t
            if info["fell_at"] is not None and t > info["fell_at"] + 1.0:
                break
            if k % every:
                continue
            n = int(round(1.4 / dt))
            v = path[-1] - path[max(0, len(path) - n)]
            walk = v / np.linalg.norm(v) if np.linalg.norm(v) > 0.05 else np.array([1.0, 0.0])
            R = d.xmat[pel].reshape(3, 3)
            fwd = R[:2, 0] / np.linalg.norm(R[:2, 0])
            yaw = np.degrees(np.arctan2(walk[0] * fwd[1] - walk[1] * fwd[0], walk @ fwd))
            yaws.append(yaw)
            base = d.xpos[pel].copy()
            base[2] += 0.85                                         # just above the head, visible from above
            cam.arrows = [(base, base + 0.6 * np.r_[walk, 0.0], (0.1, 0.75, 0.2, 1.0)),
                          (base, base + 0.6 * np.r_[fwd, 0.0], (0.9, 0.1, 0.1, 1.0))]
            img = Image.fromarray(cam.render(follow=1.0))
            dr = ImageDraw.Draw(img)
            rec = np.array(yaws[-int(1.5 * FPS):])
            dr.rectangle([0, 0, w, 56], fill=(255, 255, 255))
            dr.text((8, 4), label, fill=(20, 20, 20), font=_font(15, True))
            dr.text((8, 26), f"pelvis vs walking direction {yaw:+5.1f} deg   (last 1.5 s range "
                             f"{np.ptp(rec):4.0f} deg; people ~10-15)", fill=(60, 60, 60), font=_font(13))
            dr.rectangle([0, h - 28, w, h], fill=(255, 255, 255))
            dr.text((8, h - 22), f"t = {t:4.1f} s    hip rotation R/L {np.degrees(d.qpos[hr[0]]):+4.0f} / "
                                 f"{np.degrees(d.qpos[hr[1]]):+4.0f} deg", fill=(30, 30, 30), font=_font(13))
            if info["fell_at"] is not None:
                dr.text((w - 150, h - 24), f"FELL at {info['fell_at']:.1f} s", fill=(200, 30, 30), font=_font(16, True))
            frames.append(np.asarray(img))
        person.close()
    finally:
        server.terminate()
    return frames


def scenario_kinesis(out, T=12.0, w=480, h=400):
    from PIL import Image
    a = _kinesis_run("hip_abd_stiffness=0,hip_rot_stiffness=0", 7611, T, w, h,
                     "Frontal-plane hip springs relaxed: walks, pelvis twists ~50 deg")
    b = _kinesis_run("", 7612, T, w, h, "Indego hip springs on (600 / 150 N*m/rad)")
    n = max(len(a), len(b))
    a += [a[-1]] * (n - len(a))
    b += [b[-1]] * (n - len(b))
    top = _title(2 * w, "Learned human (KINESIS, muscle-driven RL policy) in the Indego, seen from above",
                 "Arrows above the head - red: pelvis heading, green: walking direction. People: ~10-15 deg of twist")
    frames = []
    for fa, fb in zip(a, b):
        frame = Image.new("RGB", (2 * w, top.height + h), (255, 255, 255))
        frame.paste(top, (0, 0))
        frame.paste(Image.fromarray(fa), (0, top.height))
        frame.paste(Image.fromarray(fb), (w, top.height))
        frames.append(np.asarray(frame))
    _save(frames, os.path.join(out, "03_learned_human_pelvis_twist"))


# --------------------------------------------------------------------------------------------------- figures

def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # baselines: falls, exo work, human share, gait deviation per controller and deficit
    rows = json.load(open(os.path.join(RES, "baselines.json")))
    ctrls = [("zero", "zero torque"), ("transparent", "transparent"), ("fsm", "Indego-like FSM"),
             ("aan", "AAN to a normative template"), ("paan", "personal AAN")]
    defs = [("fresh", None, "fresh"), ("drive", 0.6, "drive 0.6"), ("drive", 0.4, "drive 0.4"),
            ("drive", 0.3, "drive 0.3"), ("fatigue", 30.0, "fatigue 30x"), ("prefatigue", 0.5, "pre-fatigue 0.5")]

    def cell(c, k, v, key):
        g = [r for r in rows if r["controller"] == c and r["kind"] == k and r["level"] == v]
        if key == "falls":
            return sum(r["fell"] for r in g), len(g)
        ok = [r[key] for r in g if not r["fell"] and r.get(key) is not None and np.isfinite(r[key])]
        return float(np.mean(ok)) if ok else np.nan

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    falls = np.array([[cell(c, k, v, "falls")[0] for k, v, _ in defs] for c, _ in ctrls])
    axs[0].imshow(falls, cmap="Reds", vmin=0, vmax=4)
    for i in range(len(ctrls)):
        for j in range(len(defs)):
            axs[0].text(j, i, f"{falls[i, j]}/4", ha="center", va="center", fontsize=10)
    axs[0].set_xticks(range(len(defs)), [d[2] for d in defs], rotation=30, ha="right")
    axs[0].set_yticks(range(len(ctrls)), [c[1] for c in ctrls])
    axs[0].set_title("Falls (4 seeds, 30 s)")
    x = np.arange(len(defs))
    for i, (c, lab) in enumerate([ctrls[0], ctrls[1], ctrls[4]]):      # the FSM and normative AAN always fall
        axs[1].bar(x + 0.27 * (i - 1), [cell(c, k, v, "exo_pos_work_per_s") for k, v, _ in defs], 0.27, label=lab)
        axs[2].bar(x + 0.27 * (i - 1), [cell(c, k, v, "gait_deviation") for k, v, _ in defs], 0.27, label=lab)
    axs[1].set_title("Exo positive work [W] (runs without a fall)")
    axs[2].set_title("Gait deviation [fresh SDs] (runs without a fall)")
    for ax in axs[1:]:
        ax.set_xticks(x, [d[2] for d in defs], rotation=30, ha="right")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Existing controllers vs the personal AAN, simulated person walking on their own (T12 AIS D)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_controllers_vs_deficits.png"), dpi=130)
    plt.close(fig)

    # personal AAN replayed on real Indego logs
    p = json.load(open(os.path.join(RES, "paan_real_replay.json")))["summary"]
    keys = [("held_out_zero_torque", "fresh walking\n(held out)"), ("zero_torque_after_ver", "zero torque\nafter VER block"),
            ("during_ver", "during the VER\ncontroller")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (m, lab) in zip(axs, [("phase_err_median_pct", "phase error, median [% of stride]"),
                                  ("mean_abs_tau_Nm", "mean |commanded torque| [N*m]"),
                                  ("frac_time_k_above_0.1", "share of time with gains > 0.1")]):
        ax.bar([k[1] for k in keys], [p[k[0]][m] for k in keys], color=["tab:green", "tab:orange", "tab:red"])
        ax.set_title(lab, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Personal AAN replayed open-loop on real Indego logs (Waterloo data: 9 adults x 3 speeds; reference "
                 "from each person's first 50 s)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_personal_aan_on_real_logs.png"), dpi=130)
    plt.close(fig)

    # learned human: pelvis twist, deficits, long walks, KIT test across rounds
    k = json.load(open(os.path.join(RES, "kinesis_deep_eval.json")))
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    ph = k["pelvis_hip_ranges"]
    names = list(ph)
    for i, (key, lab) in enumerate([("pelvis_yaw", "pelvis yaw (twist)"), ("pelvis_pitch", "pelvis pitch"),
                                    ("pelvis_roll", "pelvis roll"), ("hip_rotation", "hip rotation")]):
        vals = [ph[n][key] if ph[n][key] is not None else np.nan for n in names]
        axs[0, 0].bar(np.arange(len(names)) + 0.2 * (i - 1.5), vals, 0.2, label=lab)
    axs[0, 0].axhspan(10, 15, color="gray", alpha=0.2, label="people: pelvis yaw / hip rotation")
    short = ["reference\n(mocap)", "pretrained\nKINESIS sim", "round 2\nKINESIS sim", "round 2\nindego_sim\nno exo",
             "round 2\nindego_sim\nexo, springs\nrelaxed", "round 2\nindego_sim\nexo, springs on\n(falls at 5 s)"]
    axs[0, 0].set_xticks(range(len(names)), short if len(short) == len(names) else names, fontsize=8)
    axs[0, 0].set_ylabel("range per stride [deg]")
    axs[0, 0].set_title("The learned human twists its pelvis ~6x more than its reference motion")
    axs[0, 0].legend(fontsize=8)
    dd = k["deficit_seconds_walked_no_exo"]
    x = np.arange(len(dd["conditions"]))
    axs[0, 1].bar(x - 0.2, dd["pretrained"], 0.4, label="pretrained KINESIS")
    axs[0, 1].bar(x + 0.2, dd["round 2"], 0.4, label="round 2 (fine-tuned with the exo)")
    axs[0, 1].set_xticks(x, dd["conditions"], rotation=30, ha="right")
    axs[0, 1].set_ylabel("seconds walked (of 40)")
    axs[0, 1].set_title("Weakness tolerance without the exo (KINESIS's simulation, 0.83 m/s)")
    axs[0, 1].legend(fontsize=8)
    lr = k["long_references_round2"]
    axs[1, 0].bar(lr["conditions"], lr["walked_whole_65s_of_18"], color="tab:purple")
    for i, v in enumerate(lr["walked_whole_65s_of_18"]):
        axs[1, 0].text(i, v + 0.2, f"{v}/18", ha="center")
    axs[1, 0].set_ylim(0, 19)
    axs[1, 0].set_title("Round 2 on 18 minute-long references (0.5-1.0 m/s): walked the whole 65 s")
    axs[1, 0].tick_params(axis="x", rotation=20)
    kt = k["kit_test_success_percent"]
    x = np.arange(len(kt["conditions"]))
    for i, r in enumerate(("pretrained", "round 1", "round 2")):
        axs[1, 1].bar(x + 0.27 * (i - 1), [v if v is not None else np.nan for v in kt[r]], 0.27, label=r)
    axs[1, 1].set_xticks(x, kt["conditions"], rotation=20)
    axs[1, 1].set_ylabel("success on 108 KIT test clips [%]")
    axs[1, 1].set_title("Fine-tuning with the exo folded into KINESIS's model")
    axs[1, 1].legend(fontsize=8)
    for ax in axs.ravel():
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_learned_human.png"), dpi=130)
    plt.close(fig)

    # capacity estimate and drive vs fatigue (existing figure + classifier summary)
    import shutil
    if os.path.abspath(out).startswith(os.path.abspath(ROOT)):
        print("figures ->", out)                             # inside the repository: the other figures are in results/
        return
    for src, dst in (("deficit_estimate.png", "fig4_capacity_estimate.png"),
                     (os.path.join("waterloo", "waterloo_vs_sim.png"), "fig5_real_indego_users_vs_handtuned_sim.png"),
                     (os.path.join("waterloo", "waterloo_vs_trials.png"), "fig6_real_indego_users_vs_learned_human.png")):
        if os.path.exists(os.path.join(RES, src)):
            shutil.copy(os.path.join(RES, src), os.path.join(out, dst))
    print("figures ->", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    out = os.path.abspath(args.pop(0)) if args and args[0] not in ("aan", "estimate", "kinesis", "figures") else         os.path.join(ROOT, "docs", "media")
    what = args or ["aan", "estimate"]
    os.makedirs(out, exist_ok=True)
    from indego_aan.resources import single_thread_env
    single_thread_env()
    for w in what:
        {"aan": scenario_aan, "estimate": scenario_estimate, "kinesis": scenario_kinesis, "figures": figures}[w](out)
