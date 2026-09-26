import csv
import gc
import json
import math
import os
import platform
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from equations import LAYERS, max_batch
from models import make_model

BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEED = 2026

COLUMNS = ["S", "B", "is_validation", "status", "latency_s", "latency_p10_s", "latency_p90_s",
           "n_latency", "memory_bytes", "mem_before_bytes", "energy_J", "avg_power_W", "n_energy",
           "flops_counted", "sm_clock_MHz", "temp_C"]


def make_grid(seed=SEED):
    rng = np.random.default_rng(seed)
    extra_s = sorted(rng.choice([s for s in range(32, 513, 16) if s not in BASE_S], 4, replace=False).tolist())
    extra_b = sorted(rng.choice([b for b in range(1, 257) if b & (b - 1)], 3, replace=False).tolist())
    grid = []
    for s in sorted(BASE_S + extra_s):
        for b in sorted(BASE_B + extra_b):
            grid.append((s, b, s in extra_s or b in extra_b))
    rng.shuffle(grid)
    return grid, extra_s, extra_b


def init_nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        return pynvml, handle
    except Exception as e:
        print("Счётчик энергии недоступен, энергию не меряем:", e)
        return None


def energy_joules(nvml):
    pynvml, handle = nvml
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000


def idle_power(nvml, seconds=3.0):
    pynvml, handle = nvml
    values = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        values.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000)
        time.sleep(0.02)
    return float(np.median(values))


def run_forward(model, x):
    with torch.inference_mode():
        model(x)


def count_flops(model_meta, s, b):
    with FlopCounterMode(display=False) as counter, torch.inference_mode():
        model_meta(torch.empty(b, 3, s, s, device="meta"))
    return counter.get_total_flops()


def time_layers(model, x, reps):
    result = []
    with torch.inference_mode():
        for _ in range(reps):
            events = [torch.cuda.Event(enable_timing=True)]
            h = x
            torch.cuda.synchronize()
            events[0].record()
            for layer in model:
                h = layer(h)
                if not isinstance(layer, nn.Flatten):
                    events.append(torch.cuda.Event(enable_timing=True))
                    events[-1].record()
            torch.cuda.synchronize()
            result.append([events[i].elapsed_time(events[i + 1]) / 1000 for i in range(len(events) - 1)])
    return np.median(result, axis=0)


def measure_one(model, model_meta, nvml, s, b, device):
    row = {"S": s, "B": b, "flops_counted": count_flops(model_meta, s, b)}
    x = torch.randn(b, 3, s, s, device=device)

    start = time.perf_counter()
    n = 0
    while n < 3 or (time.perf_counter() - start < 0.3 and n < 50):
        run_forward(model, x)
        n += 1
    torch.cuda.synchronize()
    one_pass = (time.perf_counter() - start) / n

    torch.cuda.reset_peak_memory_stats()
    row["mem_before_bytes"] = torch.cuda.memory_allocated()
    run_forward(model, x)
    torch.cuda.synchronize()
    row["memory_bytes"] = torch.cuda.max_memory_allocated()

    times = []
    for _ in range(min(200, max(10, math.ceil(1.0 / one_pass)))):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_forward(model, x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    row["latency_s"] = float(np.median(times))
    row["latency_p10_s"] = float(np.percentile(times, 10))
    row["latency_p90_s"] = float(np.percentile(times, 90))
    row["n_latency"] = len(times)

    if nvml:
        torch.cuda.synchronize()
        e0, t0, n = energy_joules(nvml), time.perf_counter(), 0
        while n < 3 or time.perf_counter() - t0 < 2.0:
            run_forward(model, x)
            n += 1
            if n % 8 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        joules, seconds = energy_joules(nvml) - e0, time.perf_counter() - t0
        pynvml, handle = nvml
        row["energy_J"] = joules / n
        row["avg_power_W"] = joules / seconds
        row["n_energy"] = n
        row["sm_clock_MHz"] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        row["temp_C"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

    layers = time_layers(model, x, 5 if row["latency_s"] < 0.1 else 3)
    return row, layers


def fits(model, s, b):
    x = None
    try:
        x = torch.randn(b, 3, s, s, device="cuda")
        run_forward(model, x)
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        del x
        gc.collect()
        torch.cuda.empty_cache()


def oom_probe(model, out):
    rows = []
    for s in (512, 384):
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        budget = free + torch.cuda.memory_allocated()
        predicted = int(max_batch(s, budget))
        lo = int(0.6 * predicted)
        hi = min(int(1.2 * predicted), (2**31 - 1) // (8 * s * s))
        lo_fits, hi_fits = fits(model, s, lo), fits(model, s, hi)
        measured = None
        if lo_fits and not hi_fits:
            while hi - lo > max(1, predicted // 200):
                mid = (lo + hi) // 2
                if fits(model, s, mid):
                    lo = mid
                else:
                    hi = mid
            measured = lo
        print(f"Граница OOM при S={s}: по формуле B_max = {predicted}, на деле {measured}")
        rows.append({"S": s, "budget_bytes": budget, "total_bytes": total, "B_max_predicted": predicted,
                     "B_max_measured": measured, "lo_fits": lo_fits, "hi_fits": hi_fits})
    with open(os.path.join(out, "oom_probe.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(out="results", device="cuda", quick=False):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    model = make_model().to(device).eval()
    model_meta = make_model().to("meta").eval()
    nvml = init_nvml()

    grid, extra_s, extra_b = make_grid()
    if quick:
        grid = [g for g in grid if g[0] <= 64 and g[1] <= 8][:6]

    run_forward(model, torch.randn(2, 3, 64, 64, device=device))
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    env = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "driver": nvml[0].nvmlSystemGetDriverVersion() if nvml else None,
        "power_limit_W": nvml[0].nvmlDeviceGetPowerManagementLimit(nvml[1]) / 1000 if nvml else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "seed": SEED,
        "random_S": extra_s,
        "random_B": extra_b,
        "idle_power_W": idle_power(nvml) if nvml else None,
        "allocated_after_warmup_bytes": torch.cuda.memory_allocated(),
    }
    with open(os.path.join(out, "env.json"), "w") as f:
        json.dump(env, f, indent=2)
    print(json.dumps(env, indent=2, ensure_ascii=False))

    meas_path = os.path.join(out, "measurements.csv")
    layers_path = os.path.join(out, "layer_times.csv")
    is_new = not os.path.exists(meas_path)
    done = set()
    if not is_new:
        with open(meas_path) as f:
            done = {(int(r["S"]), int(r["B"])) for r in csv.DictReader(f)}
    meas_file = open(meas_path, "a", newline="")
    layers_file = open(layers_path, "a", newline="")
    meas_writer = csv.DictWriter(meas_file, fieldnames=COLUMNS, restval="")
    layers_writer = csv.writer(layers_file)
    if is_new:
        meas_writer.writeheader()
        layers_writer.writerow(["S", "B", "layer_index", "layer", "time_s"])

    todo = [g for g in grid if (g[0], g[1]) not in done]
    start = time.perf_counter()
    for i, (s, b, is_val) in enumerate(todo, 1):
        gc.collect()
        torch.cuda.empty_cache()
        try:
            row, layers = measure_one(model, model_meta, nvml, s, b, device)
            row["status"] = "ok"
        except torch.cuda.OutOfMemoryError:
            row = {"S": s, "B": b, "status": "OOM", "memory_bytes": "OOM",
                   "flops_counted": count_flops(model_meta, s, b)}
            layers = None
        row["is_validation"] = int(is_val)
        meas_writer.writerow(row)
        meas_file.flush()
        if layers is not None:
            for j, t in enumerate(layers):
                layers_writer.writerow([s, b, j, LAYERS[j][0], f"{t:.9g}"])
            layers_file.flush()

        elapsed = time.perf_counter() - start
        if row["status"] == "OOM":
            print(f"[{i}/{len(todo)}] S={s} B={b}: не влезло в память ({elapsed:.0f} с)")
        else:
            print(f"[{i}/{len(todo)}] S={s} B={b}: {row['latency_s'] * 1e3:.3f} мс, "
                  f"{row['memory_bytes'] / 2**20:.1f} МиБ, {row.get('energy_J', float('nan')):.3g} Дж, "
                  f"{row.get('avg_power_W', float('nan')):.0f} Вт ({elapsed:.0f} с)", flush=True)
    meas_file.close()
    layers_file.close()

    if not quick:
        oom_probe(model, out)


if __name__ == "__main__":
    main()
