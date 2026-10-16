"""How many simulation worker processes this machine can take.

Each worker runs the full-fidelity sim (~0.6 GB resident). numpy and scipy each bundle an OpenBLAS that by default
starts one thread per logical core and reserves memory for each: ~1.6 GB committed per worker, and 16-thread BLAS
pools in every process oversubscribe the CPU (the sim's matrices are small - one thread is as fast). Scripts
therefore set the BLAS / OpenMP thread counts to 1 before numpy is imported (`single_thread_env()`, inherited by
spawned workers: ~0.65 GB committed each) and size their pools with `worker_count()`: at most one worker per
physical core (throughput flattens beyond that) and no more than the free commit memory allows (Windows does not
overcommit: past the commit limit the machine pages and stalls).
"""
import ctypes
import os
import sys

BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def single_thread_env():
    for k in BLAS_VARS:
        os.environ.setdefault(k, "1")


def commit_gb():
    """(committed, limit) in GB: memory promised to processes vs RAM + page file (Windows); (None, None) elsewhere."""
    ms = _memory_status()
    if ms is None:
        return None, None
    return (ms.ullTotalPageFile - ms.ullAvailPageFile) / 2 ** 30, ms.ullTotalPageFile / 2 ** 30


def _memory_status():
    if sys.platform != "win32":
        return None

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    ms = MS()
    ms.dwLength = ctypes.sizeof(MS)
    return ms if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)) else None


def _free_commit_gb():
    """Memory that can still be committed (RAM + page file not yet promised), GB; None if unknown."""
    if sys.platform == "win32":
        ms = _memory_status()
        return None if ms is None else ms.ullAvailPageFile / 2 ** 30
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        return None


def worker_count(n_jobs=None, requested=None, per_worker_gb=0.8, reserve_gb=4.0):
    """Workers to start: `requested` if given, else one per physical core (no more than `n_jobs`); always capped by
    the free commit memory."""
    n = requested or max(1, (os.cpu_count() or 2) // 2)
    if n_jobs:
        n = min(n, n_jobs)
    free = _free_commit_gb()
    if free is not None:
        n = min(n, max(1, int((free - reserve_gb) / per_worker_gb)))
    return max(1, n)
