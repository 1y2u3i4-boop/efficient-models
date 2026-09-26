import numpy as np

# имя, тип, FLOPs на B*S^2, FLOPs на картинку, байты на B*S^2, байты на картинку, байты весов
LAYERS = [
    ("conv1", "conv", 2352, 0, 44, 0, 18816),
    ("relu1", "mem", 0, 0, 64, 0, 0),
    ("pool", "mem", 0, 0, 40, 0, 0),
    ("conv2", "conv", 6400, 0, 24, 0, 204800),
    ("relu2", "mem", 0, 0, 32, 0, 0),
    ("conv3", "conv", 2304, 0, 24, 0, 294912),
    ("relu3", "mem", 0, 0, 16, 0, 0),
    ("conv4", "conv", 1024, 0, 24, 0, 131072),
    ("relu4", "mem", 0, 0, 32, 0, 0),
    ("conv5", "conv", 4608, 0, 20, 0, 2359296),
    ("relu5", "mem", 0, 0, 8, 0, 0),
    ("conv6", "conv", 1024, 0, 12, 0, 524288),
    ("relu6", "mem", 0, 0, 16, 0, 0),
    ("gap", "mem", 0, 0, 8, 2048, 0),
    ("fc1", "gemm", 0, 262144, 0, 3072, 525312),
    ("relu7", "mem", 0, 0, 0, 2048, 0),
    ("fc2", "gemm", 0, 51200, 0, 1424, 102800),
]
N = len(LAYERS)
F_X = np.array([l[2] for l in LAYERS], dtype=float)
F_B = np.array([l[3] for l in LAYERS], dtype=float)
M_X = np.array([l[4] for l in LAYERS], dtype=float)
M_B = np.array([l[5] for l in LAYERS], dtype=float)
W = np.array([l[6] for l in LAYERS], dtype=float)

WEIGHTS = 4161296


def _arr(image_size, batch):
    return np.asarray(image_size, dtype=float), np.asarray(batch, dtype=float)


def flops(image_size, batch):
    s, b = _arr(image_size, batch)
    return b * (17712 * s**2 + 313344)


def bytes_moved(image_size, batch):
    s, b = _arr(image_size, batch)
    return 364 * b * s**2 + 8592 * b + WEIGHTS


def memory(image_size, batch):
    s, b = _arr(image_size, batch)
    return WEIGHTS + 52 * b * s**2


def max_batch(image_size, budget):
    s = np.asarray(image_size, dtype=float)
    return np.floor((budget - WEIGHTS) / (52 * s**2))


def layer_costs(image_size, batch):
    s, b = _arr(image_size, batch)
    x, b = np.broadcast_arrays(b * s**2, b)
    shape = (N,) + (1,) * x.ndim
    f = F_X.reshape(shape) * x + F_B.reshape(shape) * b
    m = M_X.reshape(shape) * x + M_B.reshape(shape) * b + W.reshape(shape)
    return f, m


def layer_times(image_size, batch, theta):
    f, m = layer_costs(image_size, batch)
    return np.maximum(m / theta["bandwidth"], f / theta["throughput"])


def latency(image_size, batch, theta):
    g = layer_times(image_size, batch, theta)
    rest = np.cumsum(g[::-1], axis=0)[::-1]
    k = np.arange(1, N + 1).reshape((N,) + (1,) * (g.ndim - 1))
    return theta["t0"] + np.max(k * theta["tau"] + rest, axis=0)


def energy(image_size, batch, theta_energy):
    t = latency(image_size, batch, theta_energy)
    return (theta_energy["p_static"] * t
            + theta_energy["e_flop"] * flops(image_size, batch)
            + theta_energy["e_byte"] * bytes_moved(image_size, batch))


if __name__ == "__main__":
    assert F_X.sum() == 17712 and F_B.sum() == 313344
    assert M_X.sum() == 364 and M_B.sum() == 8592 and W.sum() == WEIGHTS
    theta = {"t0": 30e-6, "tau": 20e-6, "bandwidth": 250e9, "throughput": 4e12}
    for s, b in [(32, 1), (224, 1), (224, 64), (512, 256)]:
        print(f"S={s} B={b}: {flops(s, b) / 1e9:.3f} GFLOP, {memory(s, b) / 2**20:.1f} МиБ, "
              f"{latency(s, b, theta) * 1e3:.3f} мс")
