# Adapted from KINESIS (src/run.py, src/learning/experts.py), Copyright (c) 2025 Mathis Group for Computational
# Neuroscience and AI, EPFL, BSD-3-Clause. Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the copyright notices, this list of conditions and the disclaimer are
# retained (see github.com/amathislab/Kinesis/blob/main/LICENSE.md). THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT
# WARRANTY OF ANY KIND.
"""Split KINESIS's pretrained legs mixture-of-experts into three stand-alone, trainable expert policies.

The mixture's PPO update only trains the gate (its action probability is the gate's choice of expert; the experts act
deterministically), so learning a new constraint means fine-tuning the experts themselves, as KINESIS trained them:
as stochastic "lattice" policies. Each expert's normaliser, network and action layer are copied into a fresh
lattice policy (exploration noise log_std as given), with the mixture's critic as the starting value network, and
saved as <out>/expert_<i>/model.pth - a checkpoint that `src/run.py ... learning.actor_type=lattice epoch=-1
output_dir=<out>/expert_<i>` continues from. The gate is retrained on the fine-tuned experts afterwards (KINESIS's
own MoE step, `run.expert_path=<out>/`).

Run with the KINESIS environment, from the KINESIS checkout:

    python <this file> [out=data/trained_models/legs/experts_from_moe] [log_std=-1.0]
"""
import os
import sys

sys.path.append(os.getcwd())


def main(out, log_std):
    import torch
    from hydra import compose, initialize_config_dir
    from src.agents import agent_dict
    moe = torch.load("data/trained_models/legs/kinesis-moe-imitation/model.pth", map_location="cpu")
    with initialize_config_dir(config_dir=os.path.join(os.getcwd(), "cfg"), version_base=None):
        cfg = compose(config_name="config_legs", overrides=[
            "exp_name=extract", "epoch=0", "learning.actor_type=lattice", "run=eval_run_legs", "run.headless=True",
            "run.motion_file=data/kit_test_motion_dict.pkl",
            "run.initial_pose_file=data/initial_pose/legs/initial_pose_test.pkl", "no_log=True", "run.num_motions=1"])
    torch.set_default_dtype(torch.float32)
    for i in range(3):
        cfg.output_dir = os.path.join(out, f"expert_{i}")
        os.makedirs(cfg.output_dir, exist_ok=True)
        agent = agent_dict[cfg.learning.agent_name](cfg, torch.float32, torch.device("cpu"), training=True,
                                                    checkpoint_epoch=0)
        sd = agent.policy_net.state_dict()
        pre = f"experts.experts.{i}."
        for k_new, k_moe in (("norm.", "0."), ("net.", "1."), ("action_mean.", "2.")):
            for k, v in moe["policy"].items():
                if k.startswith(pre + k_moe):
                    sd[k_new + k[len(pre + k_moe):]] = v.clone()
        sd["log_std"] = torch.full_like(sd["log_std"], log_std)
        agent.policy_net.load_state_dict(sd)
        agent.value_net.load_state_dict(moe["value"])
        agent.epoch = 1                                   # a checkpoint `epoch=-1` resumes from
        torch.save(agent.get_full_state_weights(), os.path.join(cfg.output_dir, "model.pth"))
        print(f"expert {i} -> {cfg.output_dir}/model.pth", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/trained_models/legs/experts_from_moe",
         float(sys.argv[2]) if len(sys.argv) > 2 else -1.0)
