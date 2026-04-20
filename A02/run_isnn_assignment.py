import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def softplus_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def lhs_sample(n: int, d: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    """Simple Latin Hypercube Sampling in [low, high]^d."""
    u = rng.random((n, d))
    x = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(n)
        x[:, j] = (perm + u[:, j]) / n
    return low + (high - low) * x


def toy_function_1(x: np.ndarray) -> np.ndarray:
    """Eq. (12): additive split function."""
    xv = x[:, 0]
    yv = x[:, 1]
    tv = x[:, 2]
    zv = x[:, 3]
    out = np.exp(-0.5 * xv) + np.log1p(np.exp(0.4 * yv)) + np.tanh(tv) + np.sin(zv) - 0.4
    return out.reshape(-1, 1)


def toy_function_2(x: np.ndarray) -> np.ndarray:
    """Eq. (13) and (14): multiplicative split function."""
    xv = x[:, 0]
    yv = x[:, 1]
    tv = x[:, 2]
    zv = x[:, 3]
    fx = np.exp(-0.3 * xv)
    fy = (0.15 * yv) ** 2
    ft = np.tanh(0.3 * tv)
    fz = 0.2 * np.sin(0.5 * zv + 2.0) + 0.5
    out = fx * fy * fz * ft
    return out.reshape(-1, 1)


@dataclass
class DatasetSpec:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    train_n: int
    test_n: int
    train_max: float
    test_max: float


def build_dataset(spec: DatasetSpec, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = lhs_sample(spec.train_n, 4, 0.0, spec.train_max, rng)
    x_test = lhs_sample(spec.test_n, 4, 0.0, spec.test_max, rng)
    y_train = spec.fn(x_train)
    y_test = spec.fn(x_test)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }


def save_dataset_csv(data: Dict[str, np.ndarray], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = ["x", "y", "t", "z", "target"]

    def _write(path: Path, x: np.ndarray, y: np.ndarray) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for xi, yi in zip(x, y):
                writer.writerow([float(xi[0]), float(xi[1]), float(xi[2]), float(xi[3]), float(yi[0])])

    _write(out_dir / f"{prefix}_train.csv", data["x_train"], data["y_train"])
    _write(out_dir / f"{prefix}_test.csv", data["x_test"], data["y_test"])


# -----------------------------
# PyTorch models
# -----------------------------


def pos_torch(w_raw: torch.Tensor) -> torch.Tensor:
    return F.softplus(w_raw)


def init_pos_raw_torch(*shape: int) -> torch.Tensor:
    # softplus(-2) ~= 0.126, which keeps constrained branches stable at initialization.
    return -2.0 + 0.1 * torch.randn(*shape)


class ISNN1Torch(nn.Module):
    def __init__(self, width: int = 10, branch_layers: int = 2, x_layers: int = 2) -> None:
        super().__init__()
        self.width = width
        self.branch_layers = branch_layers
        self.x_layers = x_layers

        self.yy_w_raw = nn.ParameterList()
        self.yy_b = nn.ParameterList()
        self.zz_w = nn.ParameterList()
        self.zz_b = nn.ParameterList()
        self.tt_w_raw = nn.ParameterList()
        self.tt_b = nn.ParameterList()

        for i in range(branch_layers):
            in_dim = 1 if i == 0 else width
            self.yy_w_raw.append(nn.Parameter(init_pos_raw_torch(width, in_dim)))
            self.yy_b.append(nn.Parameter(torch.zeros(width)))
            self.zz_w.append(nn.Parameter(0.1 * torch.randn(width, in_dim)))
            self.zz_b.append(nn.Parameter(torch.zeros(width)))
            self.tt_w_raw.append(nn.Parameter(init_pos_raw_torch(width, in_dim)))
            self.tt_b.append(nn.Parameter(torch.zeros(width)))

        self.w_xx0 = nn.Parameter(0.1 * torch.randn(width, 1))
        self.b_x0 = nn.Parameter(torch.zeros(width))
        self.w_xy_raw = nn.Parameter(init_pos_raw_torch(width, width))
        self.w_xz = nn.Parameter(0.1 * torch.randn(width, width))
        self.w_xt_raw = nn.Parameter(init_pos_raw_torch(width, width))

        self.xx_w_raw = nn.ParameterList()
        self.xx_b = nn.ParameterList()
        for _ in range(x_layers - 1):
            self.xx_w_raw.append(nn.Parameter(init_pos_raw_torch(width, width)))
            self.xx_b.append(nn.Parameter(torch.zeros(width)))

        self.w_out_raw = nn.Parameter(init_pos_raw_torch(1, width))
        self.b_out = nn.Parameter(torch.zeros(1))

    def forward(self, x_all: torch.Tensor) -> torch.Tensor:
        x0 = x_all[:, 0:1]
        y0 = x_all[:, 1:2]
        t0 = x_all[:, 2:3]
        z0 = x_all[:, 3:4]

        y = y0
        for w_raw, b in zip(self.yy_w_raw, self.yy_b):
            y = F.softplus(y @ pos_torch(w_raw).T + b)

        z = z0
        for w, b in zip(self.zz_w, self.zz_b):
            z = torch.tanh(z @ w.T + b)

        t = t0
        for w_raw, b in zip(self.tt_w_raw, self.tt_b):
            t = F.softplus(t @ pos_torch(w_raw).T + b)

        x = F.softplus(
            x0 @ self.w_xx0.T
            + y @ pos_torch(self.w_xy_raw).T
            + z @ self.w_xz.T
            + t @ pos_torch(self.w_xt_raw).T
            + self.b_x0
        )

        for w_raw, b in zip(self.xx_w_raw, self.xx_b):
            x = F.softplus(x @ pos_torch(w_raw).T + b)

        out = x @ pos_torch(self.w_out_raw).T + self.b_out
        return out


class ISNN2Torch(nn.Module):
    def __init__(self, width: int = 15, h: int = 2) -> None:
        super().__init__()
        if h < 2:
            raise ValueError("h must be >= 2 for ISNN-2")
        self.width = width
        self.h = h

        self.yy_w_raw = nn.ParameterList()
        self.yy_b = nn.ParameterList()
        self.zz_w = nn.ParameterList()
        self.zz_b = nn.ParameterList()
        self.tt_w_raw = nn.ParameterList()
        self.tt_b = nn.ParameterList()

        for i in range(h - 1):
            in_dim = 1 if i == 0 else width
            self.yy_w_raw.append(nn.Parameter(init_pos_raw_torch(width, in_dim)))
            self.yy_b.append(nn.Parameter(torch.zeros(width)))
            self.zz_w.append(nn.Parameter(0.1 * torch.randn(width, in_dim)))
            self.zz_b.append(nn.Parameter(torch.zeros(width)))
            self.tt_w_raw.append(nn.Parameter(init_pos_raw_torch(width, in_dim)))
            self.tt_b.append(nn.Parameter(torch.zeros(width)))

        self.w_xx0 = nn.Parameter(0.1 * torch.randn(width, 1))
        self.w_xy0_raw = nn.Parameter(init_pos_raw_torch(width, 1))
        self.w_xz0 = nn.Parameter(0.1 * torch.randn(width, 1))
        self.w_xt0_raw = nn.Parameter(init_pos_raw_torch(width, 1))
        self.b_x0 = nn.Parameter(torch.zeros(width))

        self.xx_w_raw = nn.ParameterList()
        self.xx0_skip = nn.ParameterList()
        self.xy_w_raw = nn.ParameterList()
        self.xz_w = nn.ParameterList()
        self.xt_w_raw = nn.ParameterList()
        self.xx_b = nn.ParameterList()

        for _ in range(1, h):
            self.xx_w_raw.append(nn.Parameter(init_pos_raw_torch(width, width)))
            self.xx0_skip.append(nn.Parameter(0.1 * torch.randn(width, 1)))
            self.xy_w_raw.append(nn.Parameter(init_pos_raw_torch(width, width)))
            self.xz_w.append(nn.Parameter(0.1 * torch.randn(width, width)))
            self.xt_w_raw.append(nn.Parameter(init_pos_raw_torch(width, width)))
            self.xx_b.append(nn.Parameter(torch.zeros(width)))

        self.w_out_raw = nn.Parameter(init_pos_raw_torch(1, width))
        self.b_out = nn.Parameter(torch.zeros(1))

    def forward(self, x_all: torch.Tensor) -> torch.Tensor:
        x0 = x_all[:, 0:1]
        y0 = x_all[:, 1:2]
        t0 = x_all[:, 2:3]
        z0 = x_all[:, 3:4]

        ys = [y0]
        zs = [z0]
        ts = [t0]

        for i in range(self.h - 1):
            ys.append(F.softplus(ys[-1] @ pos_torch(self.yy_w_raw[i]).T + self.yy_b[i]))
            zs.append(torch.tanh(zs[-1] @ self.zz_w[i].T + self.zz_b[i]))
            ts.append(F.softplus(ts[-1] @ pos_torch(self.tt_w_raw[i]).T + self.tt_b[i]))

        x = F.softplus(
            x0 @ self.w_xx0.T
            + y0 @ pos_torch(self.w_xy0_raw).T
            + z0 @ self.w_xz0.T
            + t0 @ pos_torch(self.w_xt0_raw).T
            + self.b_x0
        )

        for i in range(1, self.h):
            x = F.softplus(
                x @ pos_torch(self.xx_w_raw[i - 1]).T
                + x0 @ self.xx0_skip[i - 1].T
                + ys[i] @ pos_torch(self.xy_w_raw[i - 1]).T
                + zs[i] @ self.xz_w[i - 1].T
                + ts[i] @ pos_torch(self.xt_w_raw[i - 1]).T
                + self.xx_b[i - 1]
            )

        out = x @ pos_torch(self.w_out_raw).T + self.b_out
        return out


# -----------------------------
# NumPy manual backprop models
# -----------------------------


class AdamNumpy:
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, p in params.items():
            g = grads[k]
            if k not in self.m:
                self.m[k] = np.zeros_like(g)
                self.v[k] = np.zeros_like(g)
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[k] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1.0 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ISNN1Numpy:
    def __init__(self, width: int = 10, branch_layers: int = 2, x_layers: int = 2, seed: int = 0) -> None:
        self.width = width
        self.branch_layers = branch_layers
        self.x_layers = x_layers
        self.rng = np.random.default_rng(seed)

        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.positive_keys = set()
        self.cache: Dict[str, object] = {}

        self._init_params()

    def _add_param(self, key: str, shape: Tuple[int, ...], positive: bool) -> None:
        mean = -2.0 if positive else 0.0
        self.params[key] = self.rng.normal(mean, 0.1, size=shape)
        self.grads[key] = np.zeros(shape, dtype=np.float64)
        if positive:
            self.positive_keys.add(key)

    def _init_params(self) -> None:
        for i in range(self.branch_layers):
            in_dim = 1 if i == 0 else self.width
            self._add_param(f"Wyy_{i}", (self.width, in_dim), positive=True)
            self._add_param(f"byy_{i}", (self.width,), positive=False)
            self._add_param(f"Wzz_{i}", (self.width, in_dim), positive=False)
            self._add_param(f"bzz_{i}", (self.width,), positive=False)
            self._add_param(f"Wtt_{i}", (self.width, in_dim), positive=True)
            self._add_param(f"btt_{i}", (self.width,), positive=False)

        self._add_param("Wxx0", (self.width, 1), positive=False)
        self._add_param("bxx0", (self.width,), positive=False)
        self._add_param("Wxy", (self.width, self.width), positive=True)
        self._add_param("Wxz", (self.width, self.width), positive=False)
        self._add_param("Wxt", (self.width, self.width), positive=True)

        for i in range(self.x_layers - 1):
            self._add_param(f"Wxx_{i}", (self.width, self.width), positive=True)
            self._add_param(f"bxx_{i}", (self.width,), positive=False)

        self._add_param("Wout", (1, self.width), positive=True)
        self._add_param("bout", (1,), positive=False)

    def _w(self, key: str) -> np.ndarray:
        if key in self.positive_keys:
            return softplus_np(self.params[key])
        return self.params[key]

    def _dw_raw(self, key: str) -> np.ndarray:
        if key in self.positive_keys:
            return sigmoid_np(self.params[key])
        return np.ones_like(self.params[key])

    def zero_grads(self) -> None:
        for k in self.grads:
            self.grads[k].fill(0.0)

    def forward(self, x_all: np.ndarray, store_cache: bool = True) -> np.ndarray:
        x0 = x_all[:, 0:1]
        y0 = x_all[:, 1:2]
        t0 = x_all[:, 2:3]
        z0 = x_all[:, 3:4]

        y_states = [y0]
        y_pres = []
        for i in range(self.branch_layers):
            pre = y_states[-1] @ self._w(f"Wyy_{i}").T + self.params[f"byy_{i}"]
            y_pres.append(pre)
            y_states.append(softplus_np(pre))

        z_states = [z0]
        z_pres = []
        for i in range(self.branch_layers):
            pre = z_states[-1] @ self._w(f"Wzz_{i}").T + self.params[f"bzz_{i}"]
            z_pres.append(pre)
            z_states.append(np.tanh(pre))

        t_states = [t0]
        t_pres = []
        for i in range(self.branch_layers):
            pre = t_states[-1] @ self._w(f"Wtt_{i}").T + self.params[f"btt_{i}"]
            t_pres.append(pre)
            t_states.append(softplus_np(pre))

        x_pre0 = (
            x0 @ self._w("Wxx0").T
            + y_states[-1] @ self._w("Wxy").T
            + z_states[-1] @ self._w("Wxz").T
            + t_states[-1] @ self._w("Wxt").T
            + self.params["bxx0"]
        )
        x_states = [softplus_np(x_pre0)]
        x_pres = [x_pre0]

        for i in range(self.x_layers - 1):
            pre = x_states[-1] @ self._w(f"Wxx_{i}").T + self.params[f"bxx_{i}"]
            x_pres.append(pre)
            x_states.append(softplus_np(pre))

        out = x_states[-1] @ self._w("Wout").T + self.params["bout"]

        if store_cache:
            self.cache = {
                "x0": x0,
                "y_states": y_states,
                "y_pres": y_pres,
                "z_states": z_states,
                "z_pres": z_pres,
                "t_states": t_states,
                "t_pres": t_pres,
                "x_states": x_states,
                "x_pres": x_pres,
            }

        return out

    def backward(self, d_out: np.ndarray) -> None:
        c = self.cache
        x0 = c["x0"]
        y_states = c["y_states"]
        y_pres = c["y_pres"]
        z_states = c["z_states"]
        z_pres = c["z_pres"]
        t_states = c["t_states"]
        t_pres = c["t_pres"]
        x_states = c["x_states"]
        x_pres = c["x_pres"]

        wout = self._w("Wout")
        self.grads["Wout"] += (d_out.T @ x_states[-1]) * self._dw_raw("Wout")
        self.grads["bout"] += d_out.sum(axis=0)
        d_x = d_out @ wout

        for i in range(self.x_layers - 2, -1, -1):
            pre = x_pres[i + 1]
            d_pre = d_x * sigmoid_np(pre)
            self.grads[f"Wxx_{i}"] += (d_pre.T @ x_states[i]) * self._dw_raw(f"Wxx_{i}")
            self.grads[f"bxx_{i}"] += d_pre.sum(axis=0)
            d_x = d_pre @ self._w(f"Wxx_{i}")

        d_pre0 = d_x * sigmoid_np(x_pres[0])
        self.grads["Wxx0"] += (d_pre0.T @ x0) * self._dw_raw("Wxx0")
        self.grads["bxx0"] += d_pre0.sum(axis=0)
        self.grads["Wxy"] += (d_pre0.T @ y_states[-1]) * self._dw_raw("Wxy")
        self.grads["Wxz"] += (d_pre0.T @ z_states[-1]) * self._dw_raw("Wxz")
        self.grads["Wxt"] += (d_pre0.T @ t_states[-1]) * self._dw_raw("Wxt")

        d_y = d_pre0 @ self._w("Wxy")
        d_z = d_pre0 @ self._w("Wxz")
        d_t = d_pre0 @ self._w("Wxt")

        for i in range(self.branch_layers - 1, -1, -1):
            d_pre = d_y * sigmoid_np(y_pres[i])
            self.grads[f"Wyy_{i}"] += (d_pre.T @ y_states[i]) * self._dw_raw(f"Wyy_{i}")
            self.grads[f"byy_{i}"] += d_pre.sum(axis=0)
            d_y = d_pre @ self._w(f"Wyy_{i}")

        for i in range(self.branch_layers - 1, -1, -1):
            z_act = np.tanh(z_pres[i])
            d_pre = d_z * (1.0 - z_act * z_act)
            self.grads[f"Wzz_{i}"] += (d_pre.T @ z_states[i]) * self._dw_raw(f"Wzz_{i}")
            self.grads[f"bzz_{i}"] += d_pre.sum(axis=0)
            d_z = d_pre @ self._w(f"Wzz_{i}")

        for i in range(self.branch_layers - 1, -1, -1):
            d_pre = d_t * sigmoid_np(t_pres[i])
            self.grads[f"Wtt_{i}"] += (d_pre.T @ t_states[i]) * self._dw_raw(f"Wtt_{i}")
            self.grads[f"btt_{i}"] += d_pre.sum(axis=0)
            d_t = d_pre @ self._w(f"Wtt_{i}")

    def loss_and_backward(self, x_all: np.ndarray, y_true: np.ndarray) -> float:
        self.zero_grads()
        pred = self.forward(x_all, store_cache=True)
        diff = pred - y_true
        loss = float(np.mean(diff * diff))
        d_out = (2.0 / x_all.shape[0]) * diff
        self.backward(d_out)
        return loss

    def predict(self, x_all: np.ndarray) -> np.ndarray:
        return self.forward(x_all, store_cache=False)


class ISNN2Numpy:
    def __init__(self, width: int = 15, h: int = 2, seed: int = 0) -> None:
        if h < 2:
            raise ValueError("h must be >= 2 for ISNN-2")
        self.width = width
        self.h = h
        self.rng = np.random.default_rng(seed)

        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.positive_keys = set()
        self.cache: Dict[str, object] = {}

        self._init_params()

    def _add_param(self, key: str, shape: Tuple[int, ...], positive: bool) -> None:
        mean = -2.0 if positive else 0.0
        self.params[key] = self.rng.normal(mean, 0.1, size=shape)
        self.grads[key] = np.zeros(shape, dtype=np.float64)
        if positive:
            self.positive_keys.add(key)

    def _init_params(self) -> None:
        for i in range(self.h - 1):
            in_dim = 1 if i == 0 else self.width
            self._add_param(f"Wyy_{i}", (self.width, in_dim), positive=True)
            self._add_param(f"byy_{i}", (self.width,), positive=False)
            self._add_param(f"Wzz_{i}", (self.width, in_dim), positive=False)
            self._add_param(f"bzz_{i}", (self.width,), positive=False)
            self._add_param(f"Wtt_{i}", (self.width, in_dim), positive=True)
            self._add_param(f"btt_{i}", (self.width,), positive=False)

        self._add_param("Wxx0", (self.width, 1), positive=False)
        self._add_param("Wxy0", (self.width, 1), positive=True)
        self._add_param("Wxz0", (self.width, 1), positive=False)
        self._add_param("Wxt0", (self.width, 1), positive=True)
        self._add_param("bxx0", (self.width,), positive=False)

        for i in range(1, self.h):
            self._add_param(f"Wxx_{i}", (self.width, self.width), positive=True)
            self._add_param(f"Wxx0s_{i}", (self.width, 1), positive=False)
            self._add_param(f"Wxy_{i}", (self.width, self.width), positive=True)
            self._add_param(f"Wxz_{i}", (self.width, self.width), positive=False)
            self._add_param(f"Wxt_{i}", (self.width, self.width), positive=True)
            self._add_param(f"bxx_{i}", (self.width,), positive=False)

        self._add_param("Wout", (1, self.width), positive=True)
        self._add_param("bout", (1,), positive=False)

    def _w(self, key: str) -> np.ndarray:
        if key in self.positive_keys:
            return softplus_np(self.params[key])
        return self.params[key]

    def _dw_raw(self, key: str) -> np.ndarray:
        if key in self.positive_keys:
            return sigmoid_np(self.params[key])
        return np.ones_like(self.params[key])

    def zero_grads(self) -> None:
        for k in self.grads:
            self.grads[k].fill(0.0)

    def forward(self, x_all: np.ndarray, store_cache: bool = True) -> np.ndarray:
        x0 = x_all[:, 0:1]
        y0 = x_all[:, 1:2]
        t0 = x_all[:, 2:3]
        z0 = x_all[:, 3:4]

        y_states = [y0]
        y_pres = []
        z_states = [z0]
        z_pres = []
        t_states = [t0]
        t_pres = []

        for i in range(self.h - 1):
            yp = y_states[-1] @ self._w(f"Wyy_{i}").T + self.params[f"byy_{i}"]
            zp = z_states[-1] @ self._w(f"Wzz_{i}").T + self.params[f"bzz_{i}"]
            tp = t_states[-1] @ self._w(f"Wtt_{i}").T + self.params[f"btt_{i}"]
            y_pres.append(yp)
            z_pres.append(zp)
            t_pres.append(tp)
            y_states.append(softplus_np(yp))
            z_states.append(np.tanh(zp))
            t_states.append(softplus_np(tp))

        x_pres: List[np.ndarray] = [None] * self.h
        x_states: List[np.ndarray] = [None] * (self.h + 1)

        x_pres[0] = (
            x0 @ self._w("Wxx0").T
            + y0 @ self._w("Wxy0").T
            + z0 @ self._w("Wxz0").T
            + t0 @ self._w("Wxt0").T
            + self.params["bxx0"]
        )
        x_states[1] = softplus_np(x_pres[0])

        for i in range(1, self.h):
            x_pres[i] = (
                x_states[i] @ self._w(f"Wxx_{i}").T
                + x0 @ self._w(f"Wxx0s_{i}").T
                + y_states[i] @ self._w(f"Wxy_{i}").T
                + z_states[i] @ self._w(f"Wxz_{i}").T
                + t_states[i] @ self._w(f"Wxt_{i}").T
                + self.params[f"bxx_{i}"]
            )
            x_states[i + 1] = softplus_np(x_pres[i])

        out = x_states[self.h] @ self._w("Wout").T + self.params["bout"]

        if store_cache:
            self.cache = {
                "x0": x0,
                "y_states": y_states,
                "y_pres": y_pres,
                "z_states": z_states,
                "z_pres": z_pres,
                "t_states": t_states,
                "t_pres": t_pres,
                "x_states": x_states,
                "x_pres": x_pres,
            }

        return out

    def backward(self, d_out: np.ndarray) -> None:
        c = self.cache
        x0 = c["x0"]
        y_states = c["y_states"]
        y_pres = c["y_pres"]
        z_states = c["z_states"]
        z_pres = c["z_pres"]
        t_states = c["t_states"]
        t_pres = c["t_pres"]
        x_states = c["x_states"]
        x_pres = c["x_pres"]

        self.grads["Wout"] += (d_out.T @ x_states[self.h]) * self._dw_raw("Wout")
        self.grads["bout"] += d_out.sum(axis=0)

        d_x_states = [None] * (self.h + 1)
        for i in range(self.h + 1):
            if x_states[i] is None:
                d_x_states[i] = None
            else:
                d_x_states[i] = np.zeros_like(x_states[i])
        d_x_states[self.h] = d_out @ self._w("Wout")

        d_y_states = [np.zeros_like(y_states[i]) for i in range(self.h)]
        d_z_states = [np.zeros_like(z_states[i]) for i in range(self.h)]
        d_t_states = [np.zeros_like(t_states[i]) for i in range(self.h)]

        for i in range(self.h - 1, 0, -1):
            d_pre = d_x_states[i + 1] * sigmoid_np(x_pres[i])

            self.grads[f"Wxx_{i}"] += (d_pre.T @ x_states[i]) * self._dw_raw(f"Wxx_{i}")
            self.grads[f"Wxx0s_{i}"] += (d_pre.T @ x0) * self._dw_raw(f"Wxx0s_{i}")
            self.grads[f"Wxy_{i}"] += (d_pre.T @ y_states[i]) * self._dw_raw(f"Wxy_{i}")
            self.grads[f"Wxz_{i}"] += (d_pre.T @ z_states[i]) * self._dw_raw(f"Wxz_{i}")
            self.grads[f"Wxt_{i}"] += (d_pre.T @ t_states[i]) * self._dw_raw(f"Wxt_{i}")
            self.grads[f"bxx_{i}"] += d_pre.sum(axis=0)

            d_x_states[i] += d_pre @ self._w(f"Wxx_{i}")
            d_y_states[i] += d_pre @ self._w(f"Wxy_{i}")
            d_z_states[i] += d_pre @ self._w(f"Wxz_{i}")
            d_t_states[i] += d_pre @ self._w(f"Wxt_{i}")

        d_pre0 = d_x_states[1] * sigmoid_np(x_pres[0])
        self.grads["Wxx0"] += (d_pre0.T @ x0) * self._dw_raw("Wxx0")
        self.grads["Wxy0"] += (d_pre0.T @ y_states[0]) * self._dw_raw("Wxy0")
        self.grads["Wxz0"] += (d_pre0.T @ z_states[0]) * self._dw_raw("Wxz0")
        self.grads["Wxt0"] += (d_pre0.T @ t_states[0]) * self._dw_raw("Wxt0")
        self.grads["bxx0"] += d_pre0.sum(axis=0)

        d_y_states[0] += d_pre0 @ self._w("Wxy0")
        d_z_states[0] += d_pre0 @ self._w("Wxz0")
        d_t_states[0] += d_pre0 @ self._w("Wxt0")

        d_from_above = np.zeros_like(y_states[-1])
        for i in range(self.h - 2, -1, -1):
            d_out_i = d_y_states[i + 1] + d_from_above
            d_pre = d_out_i * sigmoid_np(y_pres[i])
            self.grads[f"Wyy_{i}"] += (d_pre.T @ y_states[i]) * self._dw_raw(f"Wyy_{i}")
            self.grads[f"byy_{i}"] += d_pre.sum(axis=0)
            d_from_above = d_pre @ self._w(f"Wyy_{i}")

        d_from_above = np.zeros_like(z_states[-1])
        for i in range(self.h - 2, -1, -1):
            d_out_i = d_z_states[i + 1] + d_from_above
            z_act = np.tanh(z_pres[i])
            d_pre = d_out_i * (1.0 - z_act * z_act)
            self.grads[f"Wzz_{i}"] += (d_pre.T @ z_states[i]) * self._dw_raw(f"Wzz_{i}")
            self.grads[f"bzz_{i}"] += d_pre.sum(axis=0)
            d_from_above = d_pre @ self._w(f"Wzz_{i}")

        d_from_above = np.zeros_like(t_states[-1])
        for i in range(self.h - 2, -1, -1):
            d_out_i = d_t_states[i + 1] + d_from_above
            d_pre = d_out_i * sigmoid_np(t_pres[i])
            self.grads[f"Wtt_{i}"] += (d_pre.T @ t_states[i]) * self._dw_raw(f"Wtt_{i}")
            self.grads[f"btt_{i}"] += d_pre.sum(axis=0)
            d_from_above = d_pre @ self._w(f"Wtt_{i}")

    def loss_and_backward(self, x_all: np.ndarray, y_true: np.ndarray) -> float:
        self.zero_grads()
        pred = self.forward(x_all, store_cache=True)
        diff = pred - y_true
        loss = float(np.mean(diff * diff))
        d_out = (2.0 / x_all.shape[0]) * diff
        self.backward(d_out)
        return loss

    def predict(self, x_all: np.ndarray) -> np.ndarray:
        return self.forward(x_all, store_cache=False)


# -----------------------------
# Training runners
# -----------------------------


def train_torch_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Dict[str, List[float]]:
    model = model.to(device)
    xtr = torch.tensor(x_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    xte = torch.tensor(x_test, dtype=torch.float32, device=device)
    yte = torch.tensor(y_test, dtype=torch.float32, device=device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "test_loss": []}

    for _ in range(epochs):
        model.train()
        optim.zero_grad()
        pred = model(xtr)
        loss = F.mse_loss(pred, ytr)
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            test_pred = model(xte)
            test_loss = F.mse_loss(test_pred, yte)

        history["train_loss"].append(float(loss.item()))
        history["test_loss"].append(float(test_loss.item()))

    return history


def train_numpy_model(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
) -> Dict[str, List[float]]:
    opt = AdamNumpy(lr=lr)
    history = {"train_loss": [], "test_loss": []}

    for _ in range(epochs):
        train_loss = model.loss_and_backward(x_train, y_train)
        opt.step(model.params, model.grads)

        pred_test = model.predict(x_test)
        test_loss = float(np.mean((pred_test - y_test) ** 2))

        history["train_loss"].append(float(train_loss))
        history["test_loss"].append(float(test_loss))

    return history


def predict_torch(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        y = model(xt).cpu().numpy()
    return y


# -----------------------------
# Plotting
# -----------------------------


def plot_losses(dataset_name: str, histories: Dict[str, Dict[str, Dict[str, List[float]]]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    backends = ["pytorch", "numpy"]
    titles = ["PyTorch", "NumPy Manual Backprop"]

    for ax, backend, title in zip(axes, backends, titles):
        for model_name, color in [("isnn1", "tab:blue"), ("isnn2", "tab:orange")]:
            h = histories[backend][model_name]
            ax.plot(h["train_loss"], color=color, linestyle="-", label=f"{model_name.upper()} train")
            ax.plot(h["test_loss"], color=color, linestyle="--", label=f"{model_name.upper()} test")
        ax.set_yscale("log")
        ax.set_title(f"{dataset_name} - {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_behavior(
    dataset_name: str,
    y_true: np.ndarray,
    sweep: np.ndarray,
    preds: Dict[str, Dict[str, np.ndarray]],
    train_max: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    grid = [("pytorch", "isnn1"), ("pytorch", "isnn2"), ("numpy", "isnn1"), ("numpy", "isnn2")]

    for ax, (backend, model_name) in zip(axes.flat, grid):
        ax.plot(sweep, y_true.ravel(), color="black", linewidth=2.0, label="Ground truth")
        ax.plot(sweep, preds[backend][model_name].ravel(), color="tab:red", linewidth=1.8, label="Prediction")
        ax.axvspan(0.0, train_max, color="tab:green", alpha=0.12, label="Interpolation region")
        ax.axvline(train_max, color="gray", linestyle=":", linewidth=1.2)
        ax.set_title(f"{backend} - {model_name.upper()}")
        ax.set_xlabel("x = y = t = z")
        ax.set_ylabel("Response")
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"Behavioral Response - {dataset_name}", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -----------------------------
# Main experiment pipeline
# -----------------------------


def run_dataset(
    spec: DatasetSpec,
    data: Dict[str, np.ndarray],
    out_root: Path,
    epochs_torch: int,
    epochs_numpy: int,
    lr_torch: float,
    lr_numpy: float,
    seed: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    dataset_out = out_root / spec.name
    dataset_out.mkdir(parents=True, exist_ok=True)

    histories: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        "pytorch": {},
        "numpy": {},
    }

    set_seed(seed)
    torch_isnn1 = ISNN1Torch(width=10, branch_layers=2, x_layers=2)
    h_torch_1 = train_torch_model(
        torch_isnn1, x_train, y_train, x_test, y_test, epochs_torch, lr_torch, device
    )
    histories["pytorch"]["isnn1"] = h_torch_1

    set_seed(seed + 1)
    torch_isnn2 = ISNN2Torch(width=15, h=2)
    h_torch_2 = train_torch_model(
        torch_isnn2, x_train, y_train, x_test, y_test, epochs_torch, lr_torch, device
    )
    histories["pytorch"]["isnn2"] = h_torch_2

    np_isnn1 = ISNN1Numpy(width=10, branch_layers=2, x_layers=2, seed=seed)
    h_np_1 = train_numpy_model(np_isnn1, x_train, y_train, x_test, y_test, epochs_numpy, lr_numpy)
    histories["numpy"]["isnn1"] = h_np_1

    np_isnn2 = ISNN2Numpy(width=15, h=2, seed=seed + 1)
    h_np_2 = train_numpy_model(np_isnn2, x_train, y_train, x_test, y_test, epochs_numpy, lr_numpy)
    histories["numpy"]["isnn2"] = h_np_2

    plot_losses(spec.name, histories, dataset_out / "loss_curves.png")

    sweep = np.linspace(0.0, spec.test_max, 500)
    x_sweep = np.column_stack([sweep, sweep, sweep, sweep])
    y_true = spec.fn(x_sweep)

    preds = {
        "pytorch": {
            "isnn1": predict_torch(torch_isnn1, x_sweep, device),
            "isnn2": predict_torch(torch_isnn2, x_sweep, device),
        },
        "numpy": {
            "isnn1": np_isnn1.predict(x_sweep),
            "isnn2": np_isnn2.predict(x_sweep),
        },
    }

    plot_behavior(spec.name, y_true, sweep, preds, spec.train_max, dataset_out / "behavior_curves.png")

    summary_rows: List[Dict[str, object]] = []
    for backend in ["pytorch", "numpy"]:
        for model_name in ["isnn1", "isnn2"]:
            h = histories[backend][model_name]
            summary_rows.append(
                {
                    "dataset": spec.name,
                    "backend": backend,
                    "model": model_name,
                    "epochs": len(h["train_loss"]),
                    "final_train_loss": h["train_loss"][-1],
                    "final_test_loss": h["test_loss"][-1],
                }
            )

    with (dataset_out / "history.json").open("w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)

    return summary_rows


def write_summary_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "backend",
        "model",
        "epochs",
        "final_train_loss",
        "final_test_loss",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="ISNN assignment experiments")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs-torch", type=int, default=600)
    parser.add_argument("--epochs-numpy", type=int, default=600)
    parser.add_argument("--lr-torch", type=float, default=1e-3)
    parser.add_argument("--lr-numpy", type=float, default=2e-3)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--data-dir", type=str, default="generated_data")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    specs = [
        DatasetSpec(
            name="toy_eq12",
            fn=toy_function_1,
            train_n=500,
            test_n=5000,
            train_max=4.0,
            test_max=6.0,
        ),
        DatasetSpec(
            name="toy_eq13_14",
            fn=toy_function_2,
            train_n=500,
            test_n=5000,
            train_max=4.0,
            test_max=10.0,
        ),
    ]

    out_root = Path(args.output_dir)
    data_root = Path(args.data_dir)

    all_rows: List[Dict[str, object]] = []

    for i, spec in enumerate(specs):
        data = build_dataset(spec, seed=args.seed + i)
        save_dataset_csv(data, data_root, spec.name)
        rows = run_dataset(
            spec=spec,
            data=data,
            out_root=out_root,
            epochs_torch=args.epochs_torch,
            epochs_numpy=args.epochs_numpy,
            lr_torch=args.lr_torch,
            lr_numpy=args.lr_numpy,
            seed=args.seed + 10 * i,
            device=device,
        )
        all_rows.extend(rows)

    write_summary_csv(all_rows, out_root / "results_summary.csv")

    print("Finished. Artifacts generated:")
    print(f"- datasets: {data_root.resolve()}")
    print(f"- plots and metrics: {out_root.resolve()}")


if __name__ == "__main__":
    main()
