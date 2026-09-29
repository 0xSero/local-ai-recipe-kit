# E002: NVMe random-read baseline with O_DIRECT pread from a thread pool (no fio on the box).
# usage: nvme_rand.py <file> [row_bytes ...]  -> IOPS and GB/s per (block size, threads)
import os, sys, time, json, mmap, random, threading
path = sys.argv[1]; sizes = [int(x) for x in sys.argv[2:]] or [4096, 16384, 65536, 1 << 20]
fsz = os.path.getsize(path); ALIGN = 4096
def worker(bs, n, res, seed):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT); buf = mmap.mmap(-1, bs); r = random.Random(seed); done = 0
    t_end = time.perf_counter() + 4.0
    while time.perf_counter() < t_end:
        off = r.randrange(0, (fsz - bs) // ALIGN) * ALIGN; done += os.preadv(fd, [buf], off)
    res.append(done); os.close(fd)
out = {"file": path, "file_bytes": fsz, "cells": []}
for bs in sizes:
    for th in (1, 8, 32, 64):
        res = []; ts = [threading.Thread(target=worker, args=(bs, 0, res, i)) for i in range(th)]
        t = time.perf_counter(); [x.start() for x in ts]; [x.join() for x in ts]; dt = time.perf_counter() - t
        tot = sum(res); out["cells"].append({"bs": bs, "threads": th, "GBps": round(tot / dt / 1e9, 3), "kIOPS": round(tot / bs / dt / 1e3, 1)})
        print(out["cells"][-1], flush=True)
print(json.dumps(out))
