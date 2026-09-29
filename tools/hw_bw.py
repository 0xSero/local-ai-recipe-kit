# E001: host<->device bandwidth baselines (pinned/pageable H2D, D2H) and host RAM copy bandwidth.
import json, sys, time, torch
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
mod = torch.cuda if dev == "cuda" else torch.xpu
def sync(): mod.synchronize()
out = {"device": dev, "name": mod.get_device_name(0), "h2d": {}, "d2h": {}}
for mb in (1, 4, 16, 64, 256, 1024):
    n = mb << 20
    for pinned in (True, False):
        h = torch.empty(n, dtype=torch.uint8, pin_memory=pinned); h.fill_(1)
        d = torch.empty(n, dtype=torch.uint8, device=dev)
        for _ in range(3): d.copy_(h, non_blocking=True); sync()
        reps = max(3, min(50, 2048 // mb)); t = time.perf_counter()
        for _ in range(reps): d.copy_(h, non_blocking=True)
        sync(); dt = (time.perf_counter() - t) / reps
        out["h2d"][f"{mb}MB_{'pin' if pinned else 'page'}"] = round(n / dt / 1e9, 2)
        t = time.perf_counter()
        for _ in range(reps): h.copy_(d, non_blocking=True)
        sync(); dt = (time.perf_counter() - t) / reps
        out["d2h"][f"{mb}MB_{'pin' if pinned else 'page'}"] = round(n / dt / 1e9, 2)
a = torch.empty(4 << 30, dtype=torch.uint8); a.fill_(1); b = torch.empty_like(a)
t = time.perf_counter(); b.copy_(a); dt = time.perf_counter() - t
out["host_memcpy_GBps_4GB"] = round(2 * a.numel() / dt / 1e9, 2)
out["torch_threads"] = torch.get_num_threads()
print(json.dumps(out, indent=1))
