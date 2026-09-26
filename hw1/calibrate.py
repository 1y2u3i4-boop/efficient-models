import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

from equations import bytes_moved, energy, flops, latency, layer_costs, memory
from plots import make_all


def load(results):
    df = pd.read_csv(os.path.join(results, "measurements.csv"))
    df["oom"] = df["status"] == "OOM"
    for col in ["latency_s", "memory_bytes", "energy_J", "avg_power_W", "flops_counted"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["S", "B"]).reset_index(drop=True)

    layers = pd.read_csv(os.path.join(results, "layer_times.csv"))
    layers = layers.merge(df[["S", "B", "is_validation"]], on=["S", "B"])

    probe_path = os.path.join(results, "oom_probe.csv")
    probe = pd.read_csv(probe_path) if os.path.exists(probe_path) else None
    with open(os.path.join(results, "env.json")) as f:
        env = json.load(f)
    return df, layers, probe, env


def fit_bandwidth_throughput(layers):
    cal = layers[layers["is_validation"] == 0]
    f_all, m_all = layer_costs(cal["S"].values, cal["B"].values)
    idx = np.arange(len(cal))
    f = f_all[cal["layer_index"].values, idx]
    m = m_all[cal["layer_index"].values, idx]
    y = np.log(np.maximum(cal["time_s"].values, 1e-7))

    def residuals(p):
        floor, bw, perf = np.exp(p)
        return np.log(np.maximum(np.maximum(floor, m / bw), f / perf)) - y

    best = None
    for bw0 in [100e9, 250e9]:
        for perf0 in [2e12, 5e12]:
            res = least_squares(residuals, np.log([10e-6, bw0, perf0]), loss="soft_l1", f_scale=0.1)
            if best is None or res.cost < best.cost:
                best = res
    floor, bw, perf = np.exp(best.x)
    return float(floor), float(bw), float(perf)


def fit_overhead(df, bw, perf):
    cal = df[(df["is_validation"] == 0) & ~df["oom"]]
    s, b = cal["S"].values, cal["B"].values
    y = np.log(cal["latency_s"].values)

    def residuals(p):
        theta = {"t0": np.exp(p[0]), "tau": np.exp(p[1]), "bandwidth": bw, "throughput": perf}
        return np.log(latency(s, b, theta)) - y

    res = least_squares(residuals, np.log([30e-6, 15e-6]))
    return {"t0": float(np.exp(res.x[0])), "tau": float(np.exp(res.x[1])),
            "bandwidth": bw, "throughput": perf}


def fit_energy(df, theta):
    cal = df[(df["is_validation"] == 0) & ~df["oom"]].dropna(subset=["energy_J"])
    s, b, e = cal["S"].values, cal["B"].values, cal["energy_J"].values
    a = np.column_stack([latency(s, b, theta), flops(s, b), bytes_moved(s, b)])
    scale = a.max(axis=0)
    coef, _ = nnls(a / scale / e[:, None], np.ones(len(e)))
    coef = coef / scale
    return {**theta, "p_static": float(coef[0]), "e_flop": float(coef[1]), "e_byte": float(coef[2])}


def errors(df, theta, theta_e):
    ok = df[~df["oom"]]
    s, b = ok["S"].values, ok["B"].values
    val = ok["is_validation"].values == 1
    table = {}
    for name, pred, meas in [("latency", latency(s, b, theta), ok["latency_s"].values),
                             ("memory", memory(s, b), ok["memory_bytes"].values),
                             ("energy", energy(s, b, theta_e), ok["energy_J"].values),
                             ("flops", flops(s, b), ok["flops_counted"].values)]:
        err = np.abs(pred - meas) / meas * 100
        table[name] = {"calibration_mape": float(np.nanmean(err[~val])),
                       "validation_mape": float(np.nanmean(err[val])),
                       "validation_max": float(np.nanmax(err[val]))}
    return table


def main(results="results"):
    df, layers, probe, env = load(results)

    floor, bw, perf = fit_bandwidth_throughput(layers)
    theta = fit_overhead(df, bw, perf)
    theta_e = fit_energy(df, theta)

    budget = probe["budget_bytes"].iloc[0] if probe is not None else env["gpu_total_memory_bytes"]
    predicted_oom = memory(df["S"], df["B"]) > budget
    ok = df[~df["oom"]]

    out = {
        "latency": theta,
        "energy": theta_e,
        "layer_time_floor_s": floor,
        "errors_percent": errors(df, theta, theta_e),
        "oom": {"memory_budget_bytes": float(budget),
                "predicted": int(predicted_oom.sum()),
                "measured": int(df["oom"].sum()),
                "max_predicted_memory_bytes": float(memory(df["S"], df["B"]).max())},
        "corr_flops_bytes": float(np.corrcoef(flops(ok["S"], ok["B"]), bytes_moved(ok["S"], ok["B"]))[0, 1]),
    }
    with open(os.path.join(results, "theta.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    make_all(df, layers, probe, env, theta, theta_e, os.path.join(results, "figures"))


if __name__ == "__main__":
    main()
