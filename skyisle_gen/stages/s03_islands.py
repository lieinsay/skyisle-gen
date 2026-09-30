"""③ 岛群分布：密度场 → 面积加权采样 → kNN → 陆地（势力范围 × 陆地占比）→ 分类 → 纯几何候选边集。

节点 = 岛群（R10）：一个节点是一个岛群 = 一个「邑」= 一个水共同体；群内数十小岛彼此 5–15 km，
属第三层、不进管线。产物字段沿用 islands/n_islands 等旧名，语义均为「群」。
密度 = 带基线 × 板块乘子（C++ 的 stage3.cpp：汇聚脊 / 离散谷 / 走滑错断 / 热点链 / 辐合纹理）× exp(γ·残余噪声)
       × 骨架修饰（D 空域、绕道岛弧、赤道无岛核心；骨架优先于板块）。
陆地 area_km2 = 势力范围 T × 陆地占比 f（R8）；T 依赖 kNN，故 kNN 必须先于陆地计算。
候选边 = kNN 并集 + 远程边（≤ 大船航程）+ 跨赤道远征边 + 连通性回退。
由 C++ 核心（core/src/planet/stage3.cpp；行星计划 P6c，Python 参考版删于 2026-09-30，tag python-reference-final）算，四个 npz 与摘要照旧由这里写（_write）。
"""
from __future__ import annotations

import numpy as np

CLASS_NAMES = ["dense", "medium", "sparse", "isolated"]
CLASS_ZH = {"dense": "密接群岛", "medium": "中疏诸岛", "sparse": "稀疏岛链", "isolated": "孤悬散岛"}
KIND_KNN, KIND_FAR, KIND_EXPEDITION, KIND_FALLBACK = 0, 1, 2, 3


def run(ctx):
    from ..engine import core, part, planet_config, put_part
    c = core()
    I = c.planet_stage3(c.make_config(planet_config(ctx.cfg)), int(ctx.seed), part(ctx, 1), part(ctx, 2))
    put_part(ctx, 3, I)
    return _write(ctx, c.islands_arrays(I))


def _write(ctx, R: dict) -> dict:
    """写 islands / plates / cand_edges / density_grid 四个 npz、出摘要（R 里的浮点是双精度原值，这里照旧转 float32）。"""
    s = ctx.section(3)["islands"]
    scale = ctx.cfg["shared"]["scale"]
    area, territory, land_frac, arable_frac = R["area_km2"], R["territory_km2"], R["land_frac"], R["arable_frac"]
    main_area, wall, age, cls, layered = R["main_area_km2"], R["wall_m"], R["age"], R["cls"], R["layered"]
    ctx.save_npz(3, "islands", lat=R["lat"], lon=R["lon"], xyz=R["xyz"],
                 area_km2=area.astype(np.float32), height_m=R["height_m"].astype(np.float32),
                 territory_km2=territory.astype(np.float32),
                 land_frac=land_frac.astype(np.float32),
                 arable_frac=arable_frac.astype(np.float32),
                 main_frac=R["main_frac"].astype(np.float32),
                 main_area_km2=main_area.astype(np.float32),
                 wall_m=wall.astype(np.float32),
                 age=age.astype(np.float32), plate=R["plate"].astype(np.int16),
                 cls=cls, layered=layered, density_at=R["density_at"].astype(np.float32),
                 mean_nn_days=R["mean_nn_days"].astype(np.float32))
    ctx.save_npz(3, "plates", lats=R["plates_lats"], lons=R["plates_lons"],
                 plate_id=R["plate_id"], btype=R["btype"],
                 boundary_kernel=R["boundary_kernel"].astype(np.float32),
                 conv_kernel=R["conv_kernel"].astype(np.float32),
                 age=R["plate_age"].astype(np.float32), factor=R["factor"].astype(np.float32),
                 seeds_xyz=R["seeds_xyz"])
    ctx.save_npz(3, "cand_edges", src=R["src"], dst=R["dst"],
                 dist_days=R["dist_days"], kind=R["kind"])
    ctx.save_npz(3, "density_grid", lats=R["plates_lats"], lons=R["plates_lons"], density=R["density"].astype(np.float32))

    share = {CLASS_NAMES[i]: round(float((cls == i).mean()), 3) for i in range(4)}
    area_by_cls = {CLASS_NAMES[i]: round(float(np.median(area[cls == i])), 1)
                   for i in range(4) if (cls == i).any()}
    f_by_cls = {CLASS_NAMES[i]: round(float(np.median(land_frac[cls == i])), 4)
                for i in range(4) if (cls == i).any()}
    arable_km2 = area * arable_frac
    # 口径自检（shared.scale）：25M km² × 0.10 × 100 人/km² ≈ 2.5 亿人。只是摘要，不进模型
    pop = float(arable_km2.sum() * float(scale["people_per_arable_km2"]))
    return {"n_islands": int(R["lat"].size), "n_edges": int(R["src"].size), "n_expedition": int(R["n_exp"]),
            "n_fallback": int(R["n_fallback"]), "n_g_chords": int(R["n_chord"]), "class_share": share,
            "layered_share": round(float(layered.mean()), 3),
            "main_area_median_km2": round(float(np.median(main_area)), 0),
            "stack_share": round(float(R["in_stack"].mean()), 3),
            "age_median": round(float(np.median(age)), 2),
            "wall_median_m": round(float(np.median(wall)), 0),
            "land_total_km2": round(float(area.sum()), 0),
            "land_target_km2": float(scale["total_land_km2"]),
            "territory_total_km2": round(float(territory.sum()), 0),
            "land_frac_f0": round(float(R["f0"]), 5),
            "land_frac_mean": round(float(area.sum() / territory.sum()), 4),
            "land_frac_capped_share": round(
                float((land_frac >= float(s["land_frac_cap"]) - 1e-6).mean()), 3),
            "land_frac_median_by_class": f_by_cls,
            "area_median_km2": round(float(np.median(area)), 1),
            "area_p95_km2": round(float(np.quantile(area, 0.95)), 1),
            "area_max_km2": round(float(area.max()), 1),
            "area_median_by_class": area_by_cls,
            "arable_frac_mean": round(float(arable_frac.mean()), 4),
            "arable_total_km2": round(float(arable_km2.sum()), 0),
            "implied_population_M": round(pop / 1e6, 1)}
