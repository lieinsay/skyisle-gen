"""⑧ 特征场扩散（docs/12 §一–§三 的实现，本生成器的核心）。

strength(t, j) = reach(t, o→j) × adopt(t, j)
  reach  = exp(−最短路 Σ[λ_t·cost_mode + (−ln perm_mode)])   —— 最大 reach 路径
  adopt  = 1 − resistance(t) × conflict(t, j)                 —— 同槽位对称不动点
同槽位归一化后是当地的比例分布（含「本地自有」行，恒 > 0：原则己）。

铁律自检：本阶段不读 height_m；文化只以连续场（share ∈ [0,1]）形式存在，无离散标签。
由 C++ 核心（core/src/planet/stage8.cpp；行星计划 P6d，Python 参考版删于 2026-09-30，tag python-reference-final）算，traits.resolved.json / reflect.json / fields.npz / iso.npz 与摘要
照旧由这里写（_write；特征表的中文名与短语按槽位序号从 slots.toml 取）。
"""
from __future__ import annotations

import numpy as np

from .. import MODES


# ---------------------------------------------------------------- run
def run(ctx):
    cfg8 = ctx.section(8)
    if cfg8.get("engine", "field") == "mc":
        raise NotImplementedError(
            "Monte Carlo 引擎为保留接口（docs/12 §七：确定性场版本已足够）。请用 engine='field'")
    from ..engine import core, part, planet_config, put_part
    cc = core()
    D = cc.planet_stage8(cc.make_config(planet_config(ctx.cfg)), int(ctx.seed), part(ctx, 3), part(ctx, 5), part(ctx, 6), part(ctx, 7))
    put_part(ctx, 8, D)
    return _write(ctx, _from_cpp(ctx, cc.diffusion_arrays(D)))


def _from_cpp(ctx, D: dict) -> dict:
    """C++ 的 ⑧ 产物 → _write 要的形：特征记录补上槽位的中文名与短语；手工特征表的覆盖项按 traits.toml 的原值（类型照旧）。"""
    slots = ctx.cfg["slots"]["slot"]
    manual = (ctx.cfg.get("traits_manual") or {}).get("trait") or []
    traits = []
    for k, t in enumerate(D["traits"]):
        slot = slots[t["slot_index"]]
        tr = {"id": t["id"], "slot": slot["id"], "slot_zh": slot["zh"], "phrase": slot["phrase"],
              "mode": t["mode"], "resistance": t["resistance"], "d_half_days": t["d_half_days"], "lambda": t["lambda"],
              "origin_node": int(t["origin_node"]), "origin": t["origin"], "kind": t["kind"],
              "origin_time": float(t["origin_time"]), "resistance_tier": t["resistance_tier"]}
        if t["kind"] == "manual":
            m = manual[k]
            tr["origin"] = str(m["origin"])
            for key in ("resistance", "d_half_days", "mode"):
                if key in m:
                    tr[key] = m[key]
        traits.append(tr)
    slot_ids = list(D["slots"])
    fp_report = {sid: dict(f) for sid, f in zip(slot_ids, D["fixed_point"])}
    for sid, f in fp_report.items():
        if not f["converged"]:
            print(f"  [s08] 警告：槽位 {sid} 不动点未收敛（{f['n_iter']} 轮）")
    reflect = {"iso_thr": D["iso_thr"], "components": [[int(x) for x in c] for c in D["reflect"]]} if D["has_reflect"] else None
    return {"traits": traits, "slots": slot_ids, "slot_mode": dict(zip(slot_ids, D["slot_mode"])),
            "lambda_ref": dict(zip(MODES, D["lambda_ref"])), "fixed_point": fp_report, "reflect": reflect,
            **{k: D[k] for k in ("reach", "strength", "adopt", "share", "conflict_by", "C", "L", "iso", "local_share")}}


def _write(ctx, R: dict) -> dict:
    """写 traits.resolved.json / reflect.json / fields.npz / iso.npz、出摘要（R 里的浮点是双精度原值，这里照旧转 float32）。"""
    traits, slot_ids, fp_report, lam_ref, reach = R["traits"], R["slots"], R["fixed_point"], R["lambda_ref"], R["reach"]
    T = len(traits)
    if R["reflect"] is not None:
        iso_daily = R["iso"][MODES.index("daily")]
        ctx.save_json(8, "reflect", {
            "note": "反射型障碍对象：高隔离（iso_daily ≥ 阈值）连通分量。文化积累不外流（docs/12 §四）",
            "iso_thr": R["reflect"]["iso_thr"],
            "components": [{"nodes": m, "n": len(m),
                            "max_iso": round(float(iso_daily[np.array(m)].max()), 3)}
                           for m in R["reflect"]["components"]],
        })
    for ti, t in enumerate(traits):
        t["index"] = ti
    ctx.save_json(8, "traits.resolved", {"traits": traits, "slots": slot_ids,
                                         "slot_mode": R["slot_mode"],
                                         "lambda_ref": {m: round(lam_ref[m], 6) for m in MODES},
                                         "fixed_point": fp_report})
    ctx.save_npz(8, "fields", reach=reach.astype(np.float32),
                 strength=R["strength"].astype(np.float32), adopt=R["adopt"].astype(np.float32),
                 share=R["share"].astype(np.float32), conflict_by=R["conflict_by"].astype(np.int32),
                 C=R["C"].astype(np.float32), L=R["L"].astype(np.float32))
    ctx.save_npz(8, "iso", iso=R["iso"].astype(np.float32), local_share=R["local_share"].astype(np.float32))
    kinds = {}
    for t in traits:
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    reach_p50 = {m: round(float(np.median(reach[[t["index"] for t in traits if t["mode"] == m]])), 4)
                 for m in MODES if any(t["mode"] == m for t in traits)}
    return {"n_traits": T, "kinds": kinds, "n_slots": len(slot_ids),
            "reach_median_by_mode": reach_p50,
            "fp_all_converged": all(v["converged"] for v in fp_report.values())}
