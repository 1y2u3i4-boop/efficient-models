import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from equations import LAYERS, energy, flops, latency, layer_costs, max_batch, memory

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
T4 = {"throughput": 8.1e12, "bandwidth": 320e9}


def save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(os.path.join(folder, name), dpi=150)
    plt.close(fig)


def points(ax, x, y, val, color):
    ax.scatter(x[~val], y[~val], color=color, s=16, zorder=3)
    ax.scatter(x[val], y[val], facecolor="white", edgecolor=color, marker="s", s=16, zorder=3)


def add_legend(ax):
    ax.scatter([], [], color="gray", s=16, label="замер, калибровка")
    ax.scatter([], [], facecolor="white", edgecolor="gray", marker="s", s=16, label="замер, валидация")
    ax.plot([], [], color="gray", label="формула")
    ax.legend(fontsize=7, loc="upper left")


def curves(fig, ax, df, column, predict, scale, by):
    groups = sorted(df[by].unique())
    norm = LogNorm(min(groups), max(groups))
    for g in groups:
        d = df[df[by] == g]
        color = plt.cm.viridis(0.9 * norm(g))
        if by == "S":
            x_line = np.geomspace(1, 256, 100)
            ax.plot(x_line, predict(g, x_line) * scale, color=color, lw=1)
            points(ax, d["B"].values, d[column].values * scale, d["is_validation"].values == 1, color)
        else:
            x_line = np.linspace(32, 512, 100)
            ax.plot(x_line, predict(x_line, g) * scale, color=color, lw=1)
            points(ax, d["S"].values, d[column].values * scale, d["is_validation"].values == 1, color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if by == "S":
        ax.set_xlabel("размер батча B")
        label = "размер картинки S"
    else:
        ax.set_xlabel("размер картинки S, пикс")
        ax.set_xticks([32, 64, 128, 256, 512], ["32", "64", "128", "256", "512"])
        label = "размер батча B"
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label=label)
    add_legend(ax)


def parity(ax, pred, meas, val, name, unit):
    points(ax, meas, pred, val, "tab:blue")
    lo, hi = min(meas.min(), pred.min()) / 1.5, max(meas.max(), pred.max()) * 1.5
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    ax.plot([lo, hi], [lo * 0.8, hi * 0.8], "k:", lw=0.6)
    ax.plot([lo, hi], [lo * 1.25, hi * 1.25], "k:", lw=0.6)
    err = np.abs(pred - meas) / meas * 100
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"измерено, {unit}")
    ax.set_ylabel(f"по формуле, {unit}")
    ax.set_title(f"{name}: ошибка {np.nanmean(err[~val]):.1f}% (калибровка), "
                 f"{np.nanmean(err[val]):.1f}% (валидация)")
    ax.text(0.97, 0.03, "пунктир: ±25%", transform=ax.transAxes, ha="right", fontsize=7)


def plot_flops(df, folder):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    curves(fig, ax, df, "flops_counted", flops, 1.0, by="B")
    ax.set_ylabel("FLOPs за проход")
    err = np.abs(flops(df["S"], df["B"]) - df["flops_counted"]) / df["flops_counted"] * 100
    ax.set_title(f"FLOPs: формула и подсчёт PyTorch (макс. расхождение {err.max():.2g}%)")
    save(fig, folder, "flops.png")


def plot_memory(df, probe, budget, folder):
    ok = df[~df["oom"]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    mib = 1 / 2**20

    curves(fig, axes[0], ok, "memory_bytes", memory, mib, by="B")
    axes[0].axhline(budget * mib, color="k", ls="--", lw=0.8)
    axes[0].text(512, budget * mib * 0.75, "доступная память GPU", fontsize=7, ha="right")
    oom = df[df["oom"]]
    if len(oom):
        axes[0].scatter(oom["S"], memory(oom["S"], oom["B"]) * mib, marker="x", color="red", s=30, zorder=4)
    axes[0].set_ylabel("пиковая память, МиБ")
    axes[0].set_title("Пиковая память")

    pred = memory(ok["S"], ok["B"])
    axes[1].scatter(pred * mib, ok["memory_bytes"] / pred, c=np.log2(ok["B"]), cmap="viridis", s=16)
    axes[1].axhline(1, color="k", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("по формуле, МиБ")
    axes[1].set_ylabel("измерено / по формуле")
    axes[1].set_title("Во сколько раз формула ошибается")

    if probe is not None:
        x = np.arange(len(probe))
        axes[2].bar(x - 0.2, max_batch(probe["S"], probe["budget_bytes"]), 0.4, label="по формуле", color="lightsteelblue")
        axes[2].bar(x + 0.2, probe["B_max_measured"], 0.4, label="на деле", color="tab:blue")
        axes[2].set_xticks(x, [f"S={s}" for s in probe["S"]])
        axes[2].set_ylabel("максимальный батч без OOM")
        axes[2].set_title("Граница OOM")
        axes[2].legend(fontsize=7)
    else:
        axes[2].axis("off")
    save(fig, folder, "memory.png")


def plot_three(df, column, predict, scale, name, unit, folder, filename):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    curves(fig, axes[0], df, column, predict, scale, by="S")
    curves(fig, axes[1], df, column, predict, scale, by="B")
    for ax in axes[:2]:
        ax.set_ylabel(f"{name}, {unit}")
    axes[0].set_title(f"{name} от B (линия на каждый S)")
    axes[1].set_title(f"{name} от S (линия на каждый B)")
    s, b = df["S"].values, df["B"].values
    parity(axes[2], predict(s, b) * scale, df[column].values * scale, df["is_validation"].values == 1, name, unit)
    save(fig, folder, filename)


def plot_surfaces(df, theta, theta_e, folder):
    ok = df[~df["oom"]]
    sg, bg = np.meshgrid(np.linspace(32, 512, 40), np.geomspace(1, 256, 40))
    items = [("задержка, мс", latency(sg, bg, theta) * 1e3, ok["latency_s"] * 1e3),
             ("память, МиБ", memory(sg, bg) / 2**20, ok["memory_bytes"] / 2**20),
             ("энергия, мДж", energy(sg, bg, theta_e) * 1e3, ok["energy_J"] * 1e3)]
    fig = plt.figure(figsize=(17, 5))
    for i, (name, z, meas) in enumerate(items, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(np.log2(sg), np.log2(bg), np.log10(z), cmap="viridis", alpha=0.55, lw=0)
        m = meas.notna()
        ax.scatter(np.log2(ok["S"][m]), np.log2(ok["B"][m]), np.log10(meas[m]), color="k", s=6)
        ax.set_xlabel("log2 S")
        ax.set_ylabel("log2 B")
        ax.set_zlabel(f"log10 ({name})")
        ax.set_title(f"{name}: поверхность по формуле, точки из замеров")
    save(fig, folder, "surfaces.png")


def regime_shares(s, b, theta):
    f, m = layer_costs(s, b)
    t_mem = m / theta["bandwidth"]
    t_cmp = f / theta["throughput"]
    g = np.maximum(t_mem, t_cmp)
    rest = np.cumsum(g[::-1], axis=0)[::-1]
    k = np.arange(1, len(LAYERS) + 1)[:, None]
    first = np.argmax(k * theta["tau"] + rest, axis=0)
    on_gpu = np.arange(len(LAYERS))[:, None] >= first
    launch = theta["t0"] + (first + 1) * theta["tau"]
    mem = (g * on_gpu * (t_mem >= t_cmp)).sum(axis=0)
    comp = (g * on_gpu * (t_mem < t_cmp)).sum(axis=0)
    total = launch + mem + comp
    return launch / total, mem / total, comp / total


def plot_regimes(df, theta, folder):
    ok = df[~df["oom"]].copy()
    ok["x"] = ok["B"] * ok["S"] ** 2
    ok = ok.sort_values("x")
    launch, mem, comp = regime_shares(ok["S"].values, ok["B"].values, theta)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].stackplot(ok["x"], launch, mem, comp, colors=["silver", "tab:blue", "tab:red"],
                      labels=["запуск ядер (CPU)", "упор в память", "упор в вычисления"])
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("B·S², пикселей в батче")
    axes[0].set_ylabel("доля времени по формуле")
    axes[0].set_title("На что уходит время")
    axes[0].legend(fontsize=7, loc="center left")

    points(axes[1], ok["x"].values, ok["latency_s"].values / ok["x"].values * 1e9,
           ok["is_validation"].values == 1, "tab:blue")
    for s in [32, 128, 512]:
        b = np.geomspace(1, 256, 100)
        axes[1].plot(b * s**2, latency(s, b, theta) / (b * s**2) * 1e9, lw=1, label=f"формула, S={s}")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("B·S², пикселей в батче")
    axes[1].set_ylabel("время на один пиксель входа, нс")
    axes[1].set_title("Время на пиксель")
    axes[1].legend(fontsize=7)
    save(fig, folder, "regimes.png")


def plot_layers(layers, theta, gpu, folder):
    f_all, m_all = layer_costs(layers["S"].values, layers["B"].values)
    idx = np.arange(len(layers))
    layers = layers.assign(F=f_all[layers["layer_index"].values, idx], M=m_all[layers["layer_index"].values, idx])
    layers["kind"] = [LAYERS[i][1] for i in layers["layer_index"]]
    busy = layers[layers["time_s"] > 5 * layers.groupby("layer_index")["time_s"].transform("min")]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    compute = busy[busy["kind"] != "mem"]
    for i in sorted(compute["layer_index"].unique()):
        d = compute[compute["layer_index"] == i]
        axes[0].scatter(d["F"] / d["M"], d["F"] / d["time_s"], s=10, label=LAYERS[i][0])
    ai = np.geomspace(1, 1000, 100)
    axes[0].plot(ai, np.minimum(theta["throughput"], theta["bandwidth"] * ai), "k-", lw=1, label="подобранный roofline")
    if "T4" in gpu:
        axes[0].plot(ai, np.minimum(T4["throughput"], T4["bandwidth"] * ai), "k--", lw=0.8, label="паспорт T4")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("арифметическая интенсивность, FLOP/байт")
    axes[0].set_ylabel("достигнуто, FLOP/с")
    axes[0].set_title("Roofline: свёртки и Linear")
    axes[0].legend(fontsize=7)

    mem = busy[busy["kind"] == "mem"]
    for i in sorted(mem["layer_index"].unique()):
        d = mem[mem["layer_index"] == i]
        axes[1].scatter(d["M"], d["M"] / d["time_s"] / 1e9, s=10, label=LAYERS[i][0])
    axes[1].axhline(theta["bandwidth"] / 1e9, color="k", lw=1, label="подобранная β")
    if "T4" in gpu:
        axes[1].axhline(T4["bandwidth"] / 1e9, color="k", ls="--", lw=0.8, label="паспорт T4")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("байт перекладывает слой")
    axes[1].set_ylabel("достигнутая скорость памяти, ГБ/с")
    axes[1].set_title("ReLU, pooling, GAP")
    axes[1].legend(fontsize=6, ncol=2)

    s_max = layers["S"].max()
    b_max = layers[layers["S"] == s_max]["B"].max()
    d = layers[(layers["S"] == s_max) & (layers["B"] == b_max)].sort_values("layer_index")
    pred = np.maximum(d["M"] / theta["bandwidth"], d["F"] / theta["throughput"])
    y = np.arange(len(d))
    axes[2].barh(y - 0.2, d["time_s"] * 1e3, 0.4, label="замер", color="tab:blue")
    axes[2].barh(y + 0.2, pred * 1e3, 0.4, label="формула", color="lightsteelblue")
    axes[2].set_yticks(y, [LAYERS[i][0] for i in d["layer_index"]])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("время, мс")
    axes[2].set_title(f"Время по слоям, S={s_max}, B={b_max}")
    axes[2].legend(fontsize=7)
    save(fig, folder, "layers.png")


def make_all(df, layers, probe, env, theta, theta_e, folder):
    os.makedirs(folder, exist_ok=True)
    ok = df[~df["oom"]]
    budget = probe["budget_bytes"].iloc[0] if probe is not None else env["gpu_total_memory_bytes"]
    plot_flops(df, folder)
    plot_memory(df, probe, budget, folder)
    plot_three(ok, "latency_s", lambda s, b: latency(s, b, theta), 1e3, "задержка", "мс", folder, "latency.png")
    plot_three(ok.dropna(subset=["energy_J"]), "energy_J", lambda s, b: energy(s, b, theta_e), 1e3,
               "энергия", "мДж", folder, "energy.png")
    plot_surfaces(df, theta, theta_e, folder)
    plot_regimes(df, theta, folder)
    plot_layers(layers, theta, env.get("gpu", ""), folder)
    print("Графики сохранены в", folder)
