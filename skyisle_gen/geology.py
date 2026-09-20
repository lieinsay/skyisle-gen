"""地质表现层（BACKLOG R4，表现层）：把 ③ 已有的静态板块格局讲成文字。

原则甲：地质是背景，不涉玩法。本模块只读 `s03 plates.npz`（板块 / 边界类型 / 边界核 / 岛龄网格）与
`islands.npz` 的 age / plate / layered，产出**描述性文本**，不进任何推导；不读 height_m。
"""
from __future__ import annotations

import numpy as np

BTYPE_ZH = {0: "汇聚", 1: "离散", 2: "走滑"}


class Geology:
    def __init__(self, ctx):
        try:
            self.g = ctx.load_npz(3, "plates")
            self.isl = ctx.load_npz(3, "islands")
            self.available = "age" in self.isl
        except FileNotFoundError:
            self.g, self.isl, self.available = None, None, False
            return
        lats, lons = self.g["lats"], self.g["lons"]
        lat, lon = self.isl["lat"], self.isl["lon"]
        ii = np.clip(np.searchsorted(lats, lat), 0, lats.size - 1)
        jj = np.clip(np.searchsorted(lons, ((lon + 180.0) % 360.0) - 180.0), 0, lons.size - 1)
        self.btype = self.g["btype"][ii, jj].astype(np.int64)          # 最近边界的类型
        self.bkern = self.g["boundary_kernel"][ii, jj].astype(np.float64)   # 离边界多近（1 = 在边界上）
        self.age = self.isl["age"].astype(np.float64)
        self.plate = self.isl["plate"].astype(np.int64)
        self.layered = self.isl["layered"].astype(bool)

    # ---- 单邑 ----
    def node_zh(self, j: int) -> str:
        if not self.available:
            return ""
        k = float(self.bkern[j])
        a = float(self.age[j])
        age_zh = "新岛" if a < 0.3 else ("中年" if a < 0.65 else "老岛")
        if k >= 0.5:
            where = f"处于板块{BTYPE_ZH[int(self.btype[j])]}带上"
        elif k >= 0.2:
            where = f"靠近板块{BTYPE_ZH[int(self.btype[j])]}带"
        else:
            where = "板块内部"
        return f"板块 #{int(self.plate[j])} · {where} · 岛龄 {a:.2f}（{age_zh}）" + ("，叠层" if self.layered[j] else "")

    # ---- 一个地区 ----
    def region_zh(self, members: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> str:
        """静态格局的一句话：跨几个板块、在什么边界上、岛链是否一头老一头新、老岛多否。"""
        if not self.available or members.size == 0:
            return ""
        m = members
        parts = []
        pids, cnt = np.unique(self.plate[m], return_counts=True)
        order = np.argsort(-cnt, kind="stable")
        main_share = cnt[order[0]] / m.size
        if pids.size >= 2 and cnt[order[1]] / m.size >= 0.2:
            parts.append(f"本区横跨板块 #{int(pids[order[0]])} 与 #{int(pids[order[1]])} 之缝")
        else:
            parts.append(f"本区多在板块 #{int(pids[order[0]])} 上（{main_share * 100:.0f}%）")
        near = self.bkern[m] >= 0.3
        if near.mean() >= 0.3:
            bt = int(np.bincount(self.btype[m][near], minlength=3).argmax())
            if bt == 0:
                parts.append("处在板块汇聚带：浮石随流汇聚嵌合，岛群密集" + ("、上下堆叠" if self.layered[m].mean() > 0.15 else ""))
            elif bt == 1:
                parts.append("处在板块离散带：岛群被拉开，多为新岛，空域宽")
            else:
                parts.append("处在板块走滑带：岛群错断成串")
        else:
            parts.append("远离板块边界，格局久已定型")
        # 岛链一头老一头新：岛龄与位置的秩相关
        if m.size >= 6 and (self.age[m].max() - self.age[m].min()) >= 0.4:
            a = self.age[m]
            best = None
            for name, coord in (("东", ((lon[m] - lon[m].mean() + 180.0) % 360.0) - 180.0), ("北", lat[m])):
                if coord.std() < 1e-6:
                    continue
                r = float(np.corrcoef(a, coord)[0, 1])
                if best is None or abs(r) > abs(best[1]):
                    best = (name, r)
            if best is not None and abs(best[1]) >= 0.6:
                old_end = best[0] if best[1] > 0 else {"东": "西", "北": "南"}[best[0]]
                parts.append(f"岛链一头老一头新（老的一头在{old_end}，是热点拖出的链）")
        med_age = float(np.median(self.age[m]))
        if med_age >= 0.65:
            parts.append("老岛居多，低而缓；满载低飞时偶尔透过云缝瞥见海面上大片惨白的礁")
        elif med_age < 0.3:
            parts.append("新岛居多，高而峭")
        return "地质（背景，不涉玩法）：" + "，".join(parts) + "。"
