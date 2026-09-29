"""营建调试台 /town.html 的接口（PLAN-TOWN 第五步）：风格清单、岛群里的聚落清单、营建一次（地面 PNG + 方案 JSON）。

GET  /api/town/styles                 内置风格（id、中文名、地区、形态算子与叫法、窗口放大系数）+ 合成地形与规模的名字
GET  /api/town/style?name=&sets=      解析后的风格参数（TOML 文本；sets 是 JSON 数组）
GET  /api/town/sites?run=&node=       岛群里的聚落（村 / 散户 / 镇 / 专业聚落 / 城；能营建的标出来——集镇与城在第四步、专业聚落在第六步）
POST /api/town/plan {source: synth | site, terrain, households, seed, lat, run, node, site, scale, half, style, operator, sets, save}
     → 地面（底图 PNG、抽稀的高程与占用栅格，base64）+ 方案（与 plan.json 同一套段）+ 地面统计；save = 1 时照命令行写整套产物
地面按来源与窗口缓存（换风格只要窗口放大系数相同就不重算），营建本身每次重跑。
"""
from __future__ import annotations

import base64
import io
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

GROUND_MAX_PX = 1600      # 底图最长边（像素）：窗口更大就隔格取
CACHE_N = 8


def _np(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"不能写成 JSON：{type(o)}")


def town_dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_np)


class TownApi:
    def __init__(self, out_root: Path, ctx_of):
        self.out_root = Path(out_root)
        self.ctx_of = ctx_of                       # run 名 → 管线上下文（server.App.ctx）
        self.grounds: OrderedDict = OrderedDict()
        self.lock = threading.Lock()               # 地面缓存
        self.plan_lock = threading.Lock()          # 一次只营建一个（C++ 里是单线程，排队比抢 CPU 好）

    # ---------------------------------------------------------------- 清单
    def styles(self) -> dict:
        from ..town import SCALES, TERRAINS
        from ..town.style import OPERATORS, list_styles, load_style, operator_name
        out = []
        for s in list_styles():
            st = load_style(s["id"])
            ops = st.get("village", {}).get("operators", {})
            out.append({"id": s["id"], "name": s["name"], "region": s["region"],
                        "window_factor": float(st.get("site", {}).get("window_factor", 1.0)),
                        "operators": [{"id": k, "name": operator_name(st, k), "weight": float(w)} for k, w in ops.items() if w > 0],
                        "sources": st["meta"].get("sources", [])})
        return {"styles": out, "terrains": list(TERRAINS), "scales": list(SCALES), "operators": OPERATORS,
                "plannable_scales": ["宅院", "小庄", "村"]}

    def style_text(self, name: str, sets: list[str]) -> str:
        from ..town.style import dump_toml, load_style
        return dump_toml(load_style(name, [s for s in sets if s.startswith("style.")]))

    def sites(self, run: str, node: int) -> dict:
        import json
        ctx = self.ctx_of(run)
        f = ctx.out_dir / "islands" / str(node) / "settlements.json"
        if not f.exists():
            raise FileNotFoundError(f"{run}/islands/{node} 还没有 settlements.json：先在岛群调试台生成这个岛群")
        S = json.loads(f.read_text(encoding="utf-8"))
        towns = {t["village"] for t in S.get("towns", []) if t.get("village") is not None}
        rows = []
        for v in S.get("villages", []):
            town = v["id"] in towns
            rows.append({"name": v["name"], "kind": "镇" if town else ("邑治" if v.get("seat") else "村"), "households": int(v["households"]),
                         "island": v["island"], "km": v["km"], "plannable": not town,
                         "note": "集镇在第四步：先按村营建它的农户" if town else ""})
        for h in S.get("hamlets", []):
            rows.append({"name": h["name"], "kind": "散户", "households": int(h["households"]), "island": h["island"], "km": h["km"],
                         "plannable": True, "note": ""})
        for s in S.get("specials", []):
            rows.append({"name": s["name"], "kind": s.get("kind", "专业聚落"), "households": int(s["households"]), "island": s["island"],
                         "km": s["km"], "plannable": False, "note": "专业聚落在第六步"})
        return {"run": run, "node": node, "city": bool(S.get("city")), "sites": rows}

    # ---------------------------------------------------------------- 地面
    def _ground(self, body: dict, cfg: dict, hf: float) -> tuple[dict, dict, object]:
        from ..town import scale_id
        from ..town.site import site_from_group, site_synth
        src = body.get("source", "synth")
        half = float(body["half"]) if body.get("half") not in (None, "", 0) else None
        tsets = tuple(s for s in body.get("sets", []) if s.startswith("town."))
        if src == "site":
            run, node, name = str(body["run"]), int(body["node"]), str(body["site"])
            scale = scale_id(body["scale"]) if body.get("scale") else None
            key = ("site", run, node, name, scale, half, hf, tsets)
        else:
            terrain, hh, seed = str(body.get("terrain", "河谷")), int(body.get("households", 40)), int(body.get("seed", 1))
            lat = float(body["lat"]) if body.get("lat") not in (None, "") else None
            scale = scale_id(body.get("scale") or "村")
            key = ("synth", terrain, hh, seed, lat, scale, half, hf, tsets)
        with self.lock:
            if key in self.grounds:
                self.grounds.move_to_end(key)
                return self.grounds[key]
        if src == "site":
            ctx = self.ctx_of(run)
            sd, meta = site_from_group(ctx, node, name, scale, cfg, half, hf)
            base = ctx.out_dir
        else:
            sd, meta = site_synth(terrain, scale, hh, cfg, seed=seed, lat_deg=lat, half_m=half, half_factor=hf)
            base = self.out_root
        val = (sd, meta, base)
        with self.lock:
            self.grounds[key] = val
            while len(self.grounds) > CACHE_N:
                self.grounds.popitem(last=False)
        return val

    @staticmethod
    def _ground_payload(sd: dict, occ) -> dict:
        import matplotlib.pyplot as plt
        from ..town.render import ground_rgb
        H, W = sd["H"], sd["W"]
        st = max(1, int(np.ceil(max(H, W) / GROUND_MAX_PX)))
        img = ground_rgb(sd, st, fade=0.2)
        buf = io.BytesIO()
        plt.imsave(buf, img, format="png")
        h = sd["height"][::st, ::st].astype(np.float64)
        fin = np.isfinite(h)
        h0 = float(np.nanmin(h)) if fin.any() else 0.0
        hq = np.where(fin, np.clip(np.round((h - h0) * 10.0), 0, 65534) + 1, 0).astype(np.uint16)   # 0 = 虚空；其余 = (高 − h0) × 10 + 1
        out = {"png": base64.b64encode(buf.getvalue()).decode("ascii"), "stride": st, "rows": int(hq.shape[0]), "cols": int(hq.shape[1]),
               "h0": h0, "height_u16": base64.b64encode(np.ascontiguousarray(hq).tobytes()).decode("ascii"),
               "water_u8": base64.b64encode(np.ascontiguousarray(sd["water"][::st, ::st]).astype(np.uint8).tobytes()).decode("ascii"),
               "extent": [sd["x0"], sd["x0"] + W * sd["res_m"], sd["y0"] - H * sd["res_m"], sd["y0"]]}
        if occ is not None:
            out["occ_u8"] = base64.b64encode(np.ascontiguousarray(occ[::st, ::st]).astype(np.uint8).tobytes()).decode("ascii")
        return out

    # ---------------------------------------------------------------- 营建
    def plan(self, body: dict) -> dict:
        from ..town import SCALE_ZH, TERRAIN_ZH, town_config
        from ..town.output import out_dir, plan_json, write_plan_json, write_site, write_style
        from ..town.plan import make_plan
        from ..town.site import site_stats
        from ..town.style import load_style, style_hash
        sets = [str(x).strip() for x in body.get("sets", []) if str(x).strip()]
        bad = [x for x in sets if not (x.startswith("style.") or x.startswith("town.")) or "=" not in x]
        if bad:
            raise ValueError(f"参数覆盖只接受 style.a.b=value 或 town.a.b=value：{bad}")
        body = {**body, "sets": sets}
        cfg = town_config(sets)
        st = load_style(body["style"], sets) if body.get("style") else None
        hf = float(st.get("site", {}).get("window_factor", 1.0)) if st else 1.0
        t0 = time.perf_counter()
        sd, meta, base = self._ground(body, cfg, hf)
        meta = dict(meta)
        t1 = time.perf_counter()
        stats = site_stats(sd)
        P, J = None, {}
        with self.plan_lock:
            if st is not None:
                P = make_plan(sd, meta, st, body.get("operator") or None)
                meta["style_name"] = st["meta"]["name"]
                J = plan_json(P, st, style_hash(st))
        t2 = time.perf_counter()
        saved = None
        if body.get("save"):
            from ..town.render import write_plan_png, write_plan_svg
            tag = (st["meta"]["name"] + (f"-{P['op_name']}" if body.get("operator") else "")) if st else None
            out = out_dir(base, meta, tag)
            write_site(out, sd, P["occ"] if P else None)
            write_plan_json(out, sd, meta, stats, extra={**J, "timing_s": {"site": round(t1 - t0, 3), "plan": round(t2 - t1, 3)}})
            write_plan_png(out, sd, meta, cfg, stats, P, st["meta"]["name"] if st else None)
            if P:
                write_plan_svg(out, sd, meta, P, st["meta"]["name"])
                write_style(out, st)
            saved = str(out)
        title = (f"{meta['run']} #{meta['node']} {meta['site']}" if meta["source"] == "island"
                 else f"合成地形「{TERRAIN_ZH.get(meta.get('terrain'), meta.get('terrain', ''))}」种子 {meta.get('seed')}")
        return {"title": title, "scale": SCALE_ZH.get(meta["scale"], meta["scale"]), "meta": meta, "stats": stats,
                "frame": {"res_m": sd["res_m"], "H": sd["H"], "W": sd["W"], "x0": sd["x0"], "y0": sd["y0"], "lat_deg": sd["lat_deg"]},
                "rivers": [{"line": np.round(np.asarray(r["line"], dtype=np.float64), 1).tolist(), "seasonal": bool(r["seasonal"])} for r in sd["rivers"]],
                "ground": self._ground_payload(sd, P["occ"] if P else None),
                "plan": J or None, "op_name": P["op_name"] if P else None, "ops_fit": list(P.get("ops_fit", [])) if P else [],
                "timing_s": {"site": round(t1 - t0, 3), "plan": round(t2 - t1, 3)}, "saved": saved}
