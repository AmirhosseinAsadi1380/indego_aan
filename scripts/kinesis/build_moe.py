"""Put fine-tuned experts back into the pretrained KINESIS mixture: the pretrained gate (which already knows how to
route between the experts), its observation normaliser and critic, with each expert's normaliser, network and action
layer replaced by the fine-tuned one. Saved as a checkpoint that gate training continues from (epoch=-1).

    python build_moe.py <pretrained mixture model.pth> <expert prefix, e.g. runs/exo_expert_> <out model.pth>
"""
import sys

import torch


def main(pretrained, prefix, out):
    moe = torch.load(pretrained, map_location="cpu")
    for i in range(3):
        expert = torch.load(f"{prefix}{i}/model.pth", map_location="cpu")["policy"]
        for k_moe, k_exp in (("0.", "norm."), ("1.", "net."), ("2.", "action_mean.")):
            pre = f"experts.experts.{i}.{k_moe}"
            keys = [k for k in moe["policy"] if k.startswith(pre)]
            for k in keys:
                moe["policy"][k] = expert[k_exp + k[len(pre):]].clone()
            print(f"expert {i}: {len(keys)} tensors from {k_exp}*")
    moe["epoch"] = 1
    moe["frame"] = 0
    torch.save(moe, out)
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
