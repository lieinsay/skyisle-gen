"""文化场的公共读取与度量：槽位份额、TV 文化距离、同言线。

文化距离（决策 7）：D(i,j) = mean_s TV(p_s(i), p_s(j))，TV = ½Σ|p−q| ∈ [0,1]，
读作「几成做法不同」。可按传播模式过滤槽位。
"""
from __future__ import annotations

import numpy as np

from . import MODES


class World:
    """惰性读取一次 run 的全部产物（viz/probe/check/ninegrid 共用，只读）。"""

    def __init__(self, ctx):
        self.ctx = ctx
        self._cache: dict = {}

    def _get(self, key, loader):
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    @property
    def islands(self):
        return self._get("islands", lambda: self.ctx.load_npz(3, "islands"))

    @property
    def cand_edges(self):
        return self._get("cand_edges", lambda: self.ctx.load_npz(3, "cand_edges"))

    @property
    def perm(self):
        return self._get("perm", lambda: self.ctx.load_npz(5, "perm"))

    @property
    def routes(self):
        return self._get("routes", lambda: self.ctx.load_npz(6, "routes"))

    @property
    def fields(self):
        return self._get("fields", lambda: self.ctx.load_npz(8, "fields"))

    @property
    def iso(self):
        return self._get("iso", lambda: self.ctx.load_npz(8, "iso"))

    @property
    def traits_meta(self):
        return self._get("traits_meta", lambda: self.ctx.load_json(8, "traits.resolved"))

    @property
    def traits(self) -> list[dict]:
        return self.traits_meta["traits"]

    @property
    def slots(self) -> list[str]:
        return self.traits_meta["slots"]

    @property
    def centers(self):
        return self._get("centers", lambda: self.ctx.load_json(7, "centers"))

    @property
    def regions(self):
        return self._get("regions", lambda: self.ctx.load_npz(7, "regions"))

    @property
    def barriers(self):
        return self._get("barriers", lambda: self.ctx.load_json(5, "barriers"))

    @property
    def hubs(self):
        return self._get("hubs", lambda: self.ctx.load_json(6, "hubs"))

    def slot_rows(self) -> dict[str, list[int]]:
        rows: dict[str, list[int]] = {s: [] for s in self.slots}
        for t in self.traits:
            rows[t["slot"]].append(t["index"])
        return rows

    def slot_mode(self, slot: str) -> str:
        return self.traits_meta["slot_mode"][slot]

    def shares(self, slot: str) -> tuple[np.ndarray, list[str]]:
        """[V+1, N]（最后一行为「本地自有」）与行标签。"""
        rows = self.slot_rows()[slot]
        si = self.slots.index(slot)
        share = self.fields["share"][rows].astype(np.float64)
        local = self.iso["local_share"][si].astype(np.float64)[None, :]
        labels = [self.traits[r]["id"] for r in rows] + ["（本地自有）"]
        return np.concatenate([share, local], axis=0), labels

    def tv_by_slot(self, i_nodes: np.ndarray, j_nodes: np.ndarray) -> dict[str, np.ndarray]:
        """每槽位的 TV 距离（向量化，i/j 为等长节点数组）。"""
        out = {}
        for slot in self.slots:
            p, _ = self.shares(slot)
            out[slot] = 0.5 * np.abs(p[:, i_nodes] - p[:, j_nodes]).sum(axis=0)
        return out

    def tv_distance(self, i_nodes, j_nodes, mode: str | None = None) -> np.ndarray:
        i_nodes = np.atleast_1d(np.asarray(i_nodes))
        j_nodes = np.atleast_1d(np.asarray(j_nodes))
        per = self.tv_by_slot(i_nodes, j_nodes)
        slots = [s for s in self.slots if mode is None or self.slot_mode(s) == mode]
        return np.mean([per[s] for s in slots], axis=0)

    def isogloss_edges(self, theta: float = 0.5) -> np.ndarray:
        """[T, E_und] bool：特征 share 在这条边上跨过 θ。"""
        src, dst = self.cand_edges["src"], self.cand_edges["dst"]
        share = self.fields["share"].astype(np.float64)
        a, b = share[:, src], share[:, dst]
        return ((a >= theta) & (b < theta)) | ((b >= theta) & (a < theta))

    def edge_max_L(self) -> np.ndarray:
        """每条无向边四模式 −ln perm 的最大有限值（不可通模式按大数截断）。"""
        p = self.perm["perm"]
        with np.errstate(divide="ignore"):
            L = np.where(p > 0, -np.log(np.maximum(p, 1e-300)), np.inf)
        return L

    def free_edges(self, cost_days: float, perm_min: float) -> np.ndarray:
        """无障碍短边：cost ≤ 阈值（双向均）且四模式通过率 ≥ 阈值。"""
        E = self.cand_edges["src"].size
        cost = self.routes["cost"]
        cmax = np.maximum(cost[:E], cost[E:])
        return (cmax <= cost_days) & (self.perm["perm"].min(axis=1) >= perm_min)
