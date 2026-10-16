"""Phase 4 teacher, tuned instead of trained: the personal assist-as-needed controller with privileged deficit
scheduling (PersonalAANParams.deficit_gain, told the true capacity), tuned by CMA-ES on the teacher reward.

Every candidate is scored on the SAME fixed training-distribution episodes (seeds 100.., the episodes of
train_teacher.py's eval_log.csv) - common random numbers: the deficit draw, which decides most falls, is the same
for all candidates, so the comparison is not swamped by it (the problem PPO had). The untuned AAN scores ~2428 on the
12 default episodes. The best candidate is then checked on other episodes (seeds 200..) against the untuned AAN.

    python scripts/tune_teacher_aan.py [generations=15] [episodes=12] [out=results/teacher_saan]
        -> <out>/cma.jsonl (every candidate), <out>/best.json, <out>/env.json (for scripts/baselines.py "teacher")
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

# name: (low, high, log scale); the untuned PersonalAAN is the start
SPACE = {"k_hip": (20.0, 150.0, False), "k_knee": (15.0, 100.0, False), "band": (0.5, 4.0, False),
         "adapt_up": (1.0, 30.0, True), "forget": (0.02, 1.0, True), "d_ratio": (0.0, 0.2, False),
         "deficit_gain": (0.0, 1.5, False)}
START = {"k_hip": 60.0, "k_knee": 40.0, "band": 2.0, "adapt_up": 6.0, "forget": 0.15, "d_ratio": 0.08,
         "deficit_gain": 0.0}


def to_unit(v):
    out = []
    for k, (lo, hi, lg) in SPACE.items():
        x = v[k]
        out.append((np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo)) if lg else (x - lo) / (hi - lo))
    return np.array(out)


def from_unit(u):
    out = {}
    for x, (k, (lo, hi, lg)) in zip(np.clip(u, 0.0, 1.0), SPACE.items()):
        out[k] = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo)))) if lg else float(lo + x * (hi - lo))
    return out


def base_params(v):
    return {"k_max": [v["k_hip"], v["k_knee"], v["k_hip"], v["k_knee"]], "band": v["band"], "adapt_up": v["adapt_up"],
            "forget": v["forget"], "d_ratio": v["d_ratio"], "deficit_gain": v["deficit_gain"]}


def episode(job):
    """Return, fell, and mean torque cost of one zero-correction episode with the given base parameters."""
    params, seed = job
    from indego_aan.env import env_from_config
    env = env_from_config({"base": "paan", "base_params": params})
    env.reset(seed=seed)
    ret, n, torque = 0.0, 0, 0.0
    while True:
        _, r, term, trunc, info = env.step(np.zeros(4))
        ret += r
        n += 1
        torque += info["torque"]
        if term or trunc:
            return ret, bool(term), torque / n


def score(pool, cands, seeds):
    jobs = [(base_params(c), s) for c in cands for s in seeds]
    res = pool.map(episode, jobs)
    out = []
    for i in range(len(cands)):
        r = res[i * len(seeds):(i + 1) * len(seeds)]
        out.append(dict(ret=float(np.mean([x[0] for x in r])), falls=int(sum(x[1] for x in r)),
                        torque=float(np.mean([x[2] for x in r]))))
    return out


if __name__ == "__main__":
    import cma
    from indego_aan.resources import single_thread_env, worker_count
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "results", "teacher_saan")
    os.makedirs(out, exist_ok=True)
    seeds = list(range(100, 100 + n_ep))
    single_thread_env()
    es = cma.CMAEvolutionStrategy(to_unit(START), 0.15, {"bounds": [0.0, 1.0], "popsize": 8, "seed": 1,
                                                          "verbose": -9})
    log = open(os.path.join(out, "cma.jsonl"), "a")
    with Pool(worker_count()) as pool:
        ref = score(pool, [START], seeds)[0]
        log.write(json.dumps(dict(gen=0, params=START, **ref)) + "\n")
        log.flush()
        print(f"untuned AAN: return {ref['ret']:.0f}, falls {ref['falls']}/{n_ep}", flush=True)
        best = dict(params=START, **ref)
        for g in range(1, gens + 1):
            t0 = time.time()
            U = es.ask()
            cands = [from_unit(u) for u in U]
            sc = score(pool, cands, seeds)
            es.tell(U, [-x["ret"] for x in sc])
            for c, x in zip(cands, sc):
                log.write(json.dumps(dict(gen=g, params=c, **x)) + "\n")
                if x["ret"] > best["ret"]:
                    best = dict(params=c, **x)
            log.flush()
            gb = max(sc, key=lambda x: x["ret"])
            print(f"gen {g}: best of generation {gb['ret']:.0f} ({gb['falls']} falls), best so far {best['ret']:.0f} "
                  f"({best['falls']} falls) [{time.time() - t0:.0f} s]", flush=True)
            json.dump(best, open(os.path.join(out, "best.json"), "w"), indent=1)
        # check on other episodes: tuned vs untuned
        check = list(range(200, 200 + n_ep))
        tuned, untuned = score(pool, [best["params"], START], check)
        print(f"other episodes (seeds {check[0]}..{check[-1]}): tuned {tuned['ret']:.0f} ({tuned['falls']} falls), "
              f"untuned {untuned['ret']:.0f} ({untuned['falls']} falls)", flush=True)
        best.update(check_tuned=tuned, check_untuned=untuned)
        json.dump(best, open(os.path.join(out, "best.json"), "w"), indent=1)
    json.dump({"base": "paan", "base_params": base_params(best["params"])}, open(os.path.join(out, "env.json"), "w"))
