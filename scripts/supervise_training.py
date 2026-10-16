"""Run the teacher training on a shared desktop without letting it take the machine's memory.

Starts `scripts/train_teacher.py <total> auto <out> resume` when there is room, stops it (the whole process tree)
when the machine's committed memory comes within `margin` GB of its limit or one training process grows past
`proc_gb`, and restarts it - with as many workers as the free memory then allows - once the memory has come back
down. Training resumes from its last save (every 250k decisions), so a pause costs at most a few minutes of work.
Ends when the training writes <out>/DONE. Log: <out>/supervisor.log.

    python scripts/supervise_training.py [total=1e7] [out=results/teacher] [margin=3.0] [proc_gb=2.0]
"""
import os
import subprocess
import sys
import time

import psutil

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from indego_aan.resources import commit_gb  # noqa: E402

CHECK_S = 30
COOL_DOWN_S = 300               # after a stop, wait at least this long before restarting
START_ROOM_GB = 6.0             # free commit needed to (re)start: the reserve + room for a few workers


def log(path, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def tree(proc):
    try:
        return [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def largest_gb(procs):
    out = 0.0
    for p in procs:
        try:
            out = max(out, p.memory_info().private / 2 ** 30)
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            pass
    return out


def stop(proc):
    procs = tree(proc)
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=30)


if __name__ == "__main__":
    total = sys.argv[1] if len(sys.argv) > 1 else "1e7"
    out = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "teacher"))
    margin = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    proc_gb = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    os.makedirs(out, exist_ok=True)
    logf, trainlog = os.path.join(out, "supervisor.log"), os.path.join(out, "train.log")
    proc, last_stop, last_report, started, crashes = None, -1e9, 0.0, 0.0, 0
    log(logf, f"supervising: total {total}, stop within {margin} GB of the commit limit or at {proc_gb} GB/process")
    while True:
        used, limit = commit_gb()
        if proc is not None and proc.poll() is not None:            # training exited on its own
            log(logf, f"training exited (code {proc.returncode})")
            if os.path.exists(os.path.join(out, "DONE")):
                log(logf, "training complete")
                break
            quick = time.time() - started < 600
            crashes = crashes + 1 if quick else 0
            if crashes >= 3:
                log(logf, f"training failed {crashes} times within minutes of starting - giving up (see {trainlog})")
                break
            proc, last_stop = None, time.time()
        if proc is None:
            if time.time() - last_stop >= COOL_DOWN_S and limit - used >= START_ROOM_GB:
                cmd = [sys.executable, os.path.join(ROOT, "scripts", "train_teacher.py"), total, "auto", out, "resume"]
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=open(trainlog, "a"), stderr=subprocess.STDOUT)
                started = time.time()
                log(logf, f"started training (pid {proc.pid}); committed {used:.1f} of {limit:.1f} GB")
        else:
            try:
                root = psutil.Process(proc.pid)
            except psutil.NoSuchProcess:                             # exited since the poll: handled next round
                time.sleep(CHECK_S)
                continue
            procs = tree(root)
            big = largest_gb(procs)
            if used > limit - margin or big > proc_gb:
                stop(root)
                proc, last_stop = None, time.time()
                log(logf, f"STOPPED training: committed {used:.1f} of {limit:.1f} GB, largest process {big:.2f} GB")
            elif time.time() - last_report > 600:
                log(logf, f"ok: committed {used:.1f} of {limit:.1f} GB, {len(procs)} processes, largest {big:.2f} GB")
                last_report = time.time()
        time.sleep(CHECK_S)
