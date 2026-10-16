"""Phase 4: train the privileged exo teacher with PPO (Stable-Baselines3, CPU, one environment per physical core,
capped by free memory - indego_aan/resources.py).

    python scripts/train_teacher.py [total=1e7] [n_envs=auto] [out=results/teacher] [resume]

Trains until `total` decisions; with "resume" it continues from <out>/model.zip if there is one (the earlier
progress.csv is kept as progress_before_<n>.csv), so a stopped run is restarted with the same command
(scripts/supervise_training.py does that automatically). Saves <out>/model.zip + vecnormalize.pkl every 250k decisions
and a checkpoint with its normaliser every 1M in <out>/checkpoints; writes <out>/DONE at the end. Progress (SB3
logger: ep_rew_mean, ep_len_mean, ...) in <out>/progress.csv. Environment settings (indego_aan.env.env_from_config):
<out>/env.json, e.g. {"base": "paan"}; written empty (defaults) if missing. PPO settings: <out>/ppo.json overrides
PPO_DEFAULTS (n_steps, batch_size, learning_rate, target_kl, log_std_init, ...) when a run starts. Every `eval_every`
decisions (and at the start) the deterministic policy replays the same EVAL_SEEDS episodes of the training
distribution; mean return and falls go to <out>/eval_log.csv - for a residual policy the first row is its base
controller, so drifting below it shows at once. Evaluate on the success criteria with scripts/baselines.py
(controller "teacher").
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


PPO_DEFAULTS = dict(n_steps=512, batch_size=2048, n_epochs=5, learning_rate=3e-4, gamma=0.995, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.0, max_grad_norm=0.5, target_kl=None, log_std_init=-2.0,
                    eval_every=1_000_000)
EVAL_SEEDS = range(100, 112)       # fixed training-distribution episodes (deficit draw + person) for eval_log.csv


def make_env(rank, cfg):
    def _f():
        import torch
        torch.set_num_threads(1)
        from indego_aan.env import env_from_config
        return env_from_config(cfg, seed=1000 + rank)
    return _f


def load_config(out, name="env.json"):
    """<out>/<name>: this run's settings (env.json is written empty - the defaults - if missing)."""
    import json
    path = os.path.join(out, name)
    if not os.path.exists(path):
        if name != "env.json":
            return {}
        with open(path, "w") as f:
            json.dump({}, f)
    with open(path) as f:
        return json.load(f)


def fixed_eval(model, venv, cfg):
    """Deterministic policy on the EVAL_SEEDS episodes: (mean return, falls, mean torque cost per step)."""
    import numpy as np
    from indego_aan.env import env_from_config
    rets, falls, torque, n = [], 0, 0.0, 0
    for seed in EVAL_SEEDS:
        env = env_from_config(cfg)
        obs, _ = env.reset(seed=seed)
        ret = 0.0
        while True:
            act, _ = model.predict(venv.normalize_obs(obs), deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            ret += r
            torque += info["torque"]
            n += 1
            if term or trunc:
                break
        rets.append(ret)
        falls += int(term)
    return float(np.mean(rets)), falls, torque / max(n, 1)


if __name__ == "__main__":
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
    from indego_aan.resources import single_thread_env, worker_count

    total = int(float(sys.argv[1])) if len(sys.argv) > 1 else int(1e7)
    single_thread_env()
    n_envs = worker_count(requested=int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "auto" else None,
                          per_worker_gb=0.7)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "results", "teacher")
    resume = len(sys.argv) > 4 and sys.argv[4] == "resume" and os.path.exists(os.path.join(out, "model.zip"))
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(2)
    cfg = load_config(out)
    hp = dict(PPO_DEFAULTS, **load_config(out, "ppo.json"))
    venv = VecMonitor(SubprocVecEnv([make_env(i, cfg) for i in range(n_envs)]))
    if resume:
        venv = VecNormalize.load(os.path.join(out, "vecnormalize.pkl"), venv)
        model = PPO.load(os.path.join(out, "model.zip"), env=venv, device="cpu")
        prog = os.path.join(out, "progress.csv")
        if os.path.exists(prog):                  # the logger would overwrite it: keep the earlier part of the curve
            os.replace(prog, os.path.join(out, f"progress_before_{model.num_timesteps}.csv"))
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=hp["gamma"], clip_obs=10.0)
        model = PPO("MlpPolicy", venv, device="cpu", seed=0,
                    **{k: hp[k] for k in ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda",
                                          "clip_range", "ent_coef", "max_grad_norm", "target_kl")},
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=hp["log_std_init"]))
    model.set_logger(configure(out, ["csv", "stdout"]))
    print(f"{n_envs} workers, from {model.num_timesteps} to {total} decisions", flush=True)

    class SaveLatest(BaseCallback):
        """Overwrite <out>/model.zip + vecnormalize.pkl every `n` decisions (what a restart resumes from)."""

        def __init__(self, n):
            super().__init__()
            self.n, self.next = n, None

        def _on_step(self):
            if self.next is None:
                self.next = (self.num_timesteps // self.n + 1) * self.n
            if self.num_timesteps >= self.next:
                venv.save(os.path.join(out, "vecnormalize.pkl"))
                model.save(os.path.join(out, "model.zip"))
                self.next += self.n
            return True

    class FixedEval(BaseCallback):
        """Every `n` decisions (and at the very start of a run): fixed_eval -> <out>/eval_log.csv."""

        def __init__(self, n):
            super().__init__()
            self.n, self.next = n, None

        def _log(self):
            ret, falls, torque = fixed_eval(model, venv, cfg)
            path = os.path.join(out, "eval_log.csv")
            new = not os.path.exists(path)
            with open(path, "a") as f:
                if new:
                    f.write("decisions,mean_return,falls,episodes,torque_cost_per_step\n")
                f.write(f"{self.num_timesteps},{ret:.1f},{falls},{len(EVAL_SEEDS)},{torque:.4f}\n")
            print(f"fixed eval at {self.num_timesteps}: return {ret:.0f}, falls {falls}/{len(EVAL_SEEDS)}", flush=True)

        def _on_training_start(self):
            if self.num_timesteps == 0:
                self._log()

        def _on_step(self):
            if self.next is None:
                self.next = (self.num_timesteps // self.n + 1) * self.n
            if self.num_timesteps >= self.next:
                self._log()
                self.next += self.n
            return True

    ckpt = CheckpointCallback(save_freq=max(1, 1_000_000 // n_envs), save_path=os.path.join(out, "checkpoints"),
                              name_prefix="teacher", save_vecnormalize=True)
    remaining = max(total - model.num_timesteps, 0)
    if remaining:
        model.learn(total_timesteps=remaining, callback=CallbackList([SaveLatest(250_000), ckpt,
                                                                      FixedEval(int(hp["eval_every"]))]),
                    reset_num_timesteps=not resume)
    model.save(os.path.join(out, "model.zip"))
    venv.save(os.path.join(out, "vecnormalize.pkl"))
    with open(os.path.join(out, "DONE"), "w") as f:
        f.write(f"{model.num_timesteps}\n")
