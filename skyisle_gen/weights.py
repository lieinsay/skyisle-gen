"""扩散权重：w_m(e) = λ_ref,m · cost_m(e) + L_m(e)，L = −ln perm。

reach = exp(−最短路) = 所有路径中 [Π perm × exp(−λ·cost)] 的最大值。
通过率为 0 的边权为 inf（该模式下不存在）。
"""
from __future__ import annotations

import math

import numpy as np

from . import MODES


def lambda_ref(cfg: dict) -> dict[str, float]:
    dh = cfg["s08"]["half_distance_days"]
    return {m: math.log(2.0) / math.sqrt(float(dh[m][0]) * float(dh[m][1])) for m in MODES}


def neg_log_perm(perm_d: np.ndarray) -> np.ndarray:
    """L[2E, 4]。perm = 0 → inf。"""
    with np.errstate(divide="ignore"):
        return np.where(perm_d > 0, -np.log(np.maximum(perm_d, 1e-300)), np.inf)


def mode_weight(cost_m: np.ndarray, L: np.ndarray, mode: str, lam: float) -> np.ndarray:
    mi = MODES.index(mode)
    return lam * cost_m[:, mi] + L[:, mi]


def load_directed(ctx):
    """读 ⑤⑥ 产物，拼出有向图数组。返回 dict。"""
    pm = ctx.load_npz(5, "perm")
    rt = ctx.load_npz(6, "routes")
    perm_d = np.concatenate([pm["perm"], pm["perm"]], axis=0)
    return {
        "src_d": rt["src_d"], "dst_d": rt["dst_d"], "und_id": rt["und_id"],
        "cost": rt["cost"], "cost_m": rt["cost_m"], "perm_d": perm_d,
        "L": neg_log_perm(perm_d),
        "flow": rt["flow"], "node_flow": rt["node_flow"],
    }
