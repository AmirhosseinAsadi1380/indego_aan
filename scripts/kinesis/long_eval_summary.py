"""Summarise a KINESIS evaluation log of a merged long-reference set (merge_motions.py): clips walked to the end and
mean share of each reference walked, overall and per target speed (long_walk_<speed>[_r<k>]; the speeds actually
walked are in long_walk_index.json).

    python <this file> <eval.log> <set>_names.json <long_walk_index.json>
"""
import json
import re
import sys
from collections import defaultdict


def main(log, names_file, index_file):
    names = json.load(open(names_file))
    index = json.load(open(index_file))
    runs = {}
    for line in open(log, errors="replace"):
        m = re.search(r"Run (\d+): (Success|Failure) on attempt \d+\. MPJPE: [\d.]+, Frame Coverage: \[?([\d.]+)", line)
        if m:
            runs[int(m.group(1))] = (m.group(2) == "Success", float(m.group(3)))
    if not runs:
        print("no evaluated clips in", log)
        return 1
    band = defaultdict(list)
    for i, (ok, cov) in runs.items():
        m = re.match(r"long_walk_([\d.]+)", names[i])                  # target speed band of the reference
        band[float(m.group(1)) if m else float("nan")].append((ok, cov))
    n_ok = sum(ok for ok, _ in runs.values())
    cov = sum(c for _, c in runs.values()) / len(runs)
    parts = [f"{b:.1f}: {sum(o for o, _ in r)}/{len(r)}" for b, r in sorted(band.items())]
    print(f"whole 65 s: {n_ok}/{len(runs)}, mean {cov:.0f} % walked | by speed (m/s) " + ", ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
