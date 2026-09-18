"""本地操作台服务（仅标准库 http.server）。

GET  /                        操作台页面
GET  /api/runs                产物目录列表
GET  /api/world?run=          世界数据包
GET  /api/fields?run=         特征场（量化）
GET  /api/grid?run=           ② 风场 / ④ 气候 1° 网格场（量化 + base64，R2）
GET  /api/texture?run=        行星贴图 PNG
GET  /api/config?run=         生效配置
GET  /api/check?run=          验收报告
GET  /api/ninegrid?run=&region=   九格表 markdown
GET  /api/path?run=&a=&b=&mode=   最优路径逐跳
GET  /api/island?run=&node=[&year=0][&force=1]   岛群生成器（第三层）：按需生成并返回摘要；/api/island/preview 取 preview.png
GET  /island.html?run=&node=[&year=]   岛群调试台（2D 图层、四季、逐日天气、改年份 / 参数重生成）
GET  /api/island/stats?run=            全量季型统计（islands/season_stats.json，没有就算，8000 群约 5 s）
GET  /api/island/data?run=&node=&year=   island.json + climate.json（含逐日天气）
GET  /api/island/raster?run=&node=       terrain.npz 的栅格（base64 定型数组，过大时抽稀）
POST /api/island/regen {run, node, year, sets:[...]}   强制重生成（可带 island.* 参数覆盖）
POST /api/run  {seed, sets:[...], base_run}   后台重跑管线
GET  /api/run/status          进度
"""
from __future__ import annotations

import json
import time
import threading
import traceback
import urllib.parse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..cli import _ctx_from_run
from .bundle import build_fields, build_grid, build_texture, build_world, dumps

STATIC = Path(__file__).parent / "static"


class App:
    def __init__(self, out_root: Path):
        self.out_root = Path(out_root)
        self.cache: dict[tuple, object] = {}
        self.lock = threading.Lock()
        self.island_lock = threading.Lock()   # 岛群生成器：一次只生成一个群，不挡住其它接口
        self.job = {"running": False, "log": [], "run_id": None, "error": None, "done_run": None}

    # ---- runs ----
    def runs(self) -> list[dict]:
        out = []
        if self.out_root.exists():
            for d in sorted(self.out_root.iterdir()):
                if (d / "manifest.json").exists() and (d / "s08_diffusion" / "fields.npz").exists():
                    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                    out.append({"run_id": d.name, "seed": m.get("seed")})
        return out

    def ctx(self, run_id: str):
        return _ctx_from_run(str(self.out_root / run_id))

    def cached(self, key: tuple, fn):
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        val = fn()
        with self.lock:
            self.cache[key] = val
        return val

    def invalidate(self, run_id: str):
        with self.lock:
            for k in [k for k in self.cache if k[1] == run_id]:
                del self.cache[k]

    # ---- pipeline job ----
    def start_run(self, seed: int, sets: list[str], run_id: str | None):
        if self.job["running"]:
            raise RuntimeError("已有任务在运行")
        from ..config import load_config
        from ..pipeline import run as pipeline_run
        cfg = load_config(sets=sets)
        rid = run_id or f"web-seed{seed}"
        cfg.setdefault("run", {})["id"] = rid
        self.job = {"running": True, "log": [], "run_id": rid, "error": None, "done_run": None}

        def work():
            try:
                pipeline_run(cfg, seed, self.out_root, upto=10, log=self.job["log"].append)
                self.invalidate(rid)
                self.job["done_run"] = rid
            except Exception as e:  # noqa: BLE001
                self.job["error"] = f"{e}\n{traceback.format_exc()[-1500:]}"
            finally:
                self.job["running"] = False

        threading.Thread(target=work, daemon=True).start()
        return rid


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *a, app: App, **kw):
        self.app = app
        super().__init__(*a, **kw)

    def log_message(self, fmt, *args):  # 安静
        pass

    def _send(self, body: bytes, ctype: str = "application/json; charset=utf-8", code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(dumps(obj).encode("utf-8"), code=code)

    def do_GET(self):
        try:
            url = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
            p = url.path
            if p in ("/", "/index.html"):
                self._send((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif p == "/island.html":
                self._send((STATIC / "island.html").read_bytes(), "text/html; charset=utf-8")
            elif p == "/api/island/stats":
                rid = q["run"]
                f = self.app.out_root / rid / "islands" / "season_stats.json"
                if not f.exists() or q.get("force") == "1":
                    from ..island import island_config
                    from ..island.climate import classify_all
                    ctx = self.app.ctx(rid)
                    with self.app.island_lock:
                        classify_all(ctx, island_config(ctx), log=lambda *a: None)
                self._send(f.read_bytes())
            elif p == "/api/island/data":
                self._json(self._island_data(q["run"], int(q["node"]), int(q.get("year", 0)), q.get("force") == "1"))
            elif p == "/api/island/raster":
                body = self._island_raster(q["run"], int(q["node"]))
                self._send(body)
            elif p.startswith("/vendor/"):
                f = (STATIC / "vendor" / Path(p).name)
                if f.exists():
                    self._send(f.read_bytes(), "application/javascript; charset=utf-8")
                else:
                    self._send(b"not found", "text/plain", 404)
            elif p == "/api/runs":
                self._json({"runs": self.app.runs(), "job": self.app.job})
            elif p == "/api/run/status":
                self._json(self.app.job)
            elif p == "/api/world":
                rid = q["run"]
                body = self.app.cached(("world", rid), lambda: dumps(build_world(self.app.ctx(rid))).encode("utf-8"))
                self._send(body)
            elif p == "/api/fields":
                rid = q["run"]
                body = self.app.cached(("fields", rid), lambda: dumps(build_fields(self.app.ctx(rid))).encode("utf-8"))
                self._send(body)
            elif p == "/api/grid":
                rid = q["run"]
                body = self.app.cached(("grid", rid), lambda: dumps(build_grid(self.app.ctx(rid))).encode("utf-8"))
                self._send(body)
            elif p == "/api/texture":
                rid = q["run"]
                body = self.app.cached(("texture", rid), lambda: build_texture(self.app.ctx(rid)))
                self._send(body, "image/png")
            elif p == "/api/config":
                rid = q["run"]
                self._json(self.app.ctx(rid).cfg)
            elif p == "/api/check":
                rid = q["run"]
                f = self.app.out_root / rid / "s10_output" / "check.json"
                self._send(f.read_bytes() if f.exists() else b'{"items":[]}')
            elif p == "/api/ninegrid":
                rid = q["run"]
                r = int(q["region"])
                ctx = self.app.ctx(rid)
                from ..ninegrid import RegionData, build_region_md
                rd = self.app.cached(("rd", rid), lambda: RegionData(ctx))
                md, _src = build_region_md(rd, r)
                self._send(md.encode("utf-8"), "text/markdown; charset=utf-8")
            elif p == "/api/path":
                rid = q["run"]
                self._json(self._path(rid, int(q["a"]), int(q["b"]), q.get("mode", "trade")))
            elif p == "/api/island":
                self._json(self._island(q["run"], int(q["node"]), int(q.get("year", 0)), q.get("force") == "1"))
            elif p == "/api/island/preview":
                f = self.app.out_root / q["run"] / "islands" / str(int(q["node"])) / ("preview_main.png" if q.get("main") == "1" else "preview.png")
                self._send(f.read_bytes(), "image/png") if f.exists() else self._send(b"not found", "text/plain", 404)
            else:
                self._send(b"not found", "text/plain", 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e), "trace": traceback.format_exc()[-1200:]}, 500)

    def _path(self, rid, a, b, mode):
        import numpy as np
        from .. import MODES
        from ..graph import CSR, dijkstra
        from ..weights import lambda_ref, load_directed
        ctx = self.app.ctx(rid)
        g = self.app.cached(("graph", rid), lambda: load_directed(ctx))
        N = int(g["src_d"].max()) + 1
        csr = self.app.cached(("csr", rid), lambda: CSR(N, g["src_d"], g["dst_d"]))
        mi = MODES.index(mode)
        lam = lambda_ref(ctx.cfg)[mode]
        wgt = lam * g["cost_m"][:, mi] + g["L"][:, mi]
        dist, pn, pe = dijkstra(csr, wgt, [a])
        if not np.isfinite(dist[b]):
            return {"reachable": False}
        path, hops = [], []
        u = b
        while u >= 0:
            path.append(int(u))
            if pn[u] >= 0:
                e = int(pe[u])
                hops.append({"node": int(u), "cost": round(float(g["cost_m"][e, mi]), 3),
                             "perm": round(float(g["perm_d"][e, mi]), 4)})
            u = int(pn[u])
        path.reverse()
        hops.reverse()
        acc = 0.0
        for h in hops:
            acc += lam * h["cost"] - np.log(max(h["perm"], 1e-300))
            h["reach"] = round(float(np.exp(-acc)), 4)
        return {"reachable": True, "path": path, "hops": hops, "reach": round(float(np.exp(-dist[b])), 4),
                "lambda_ref": lam}

    def _island(self, rid, node, year, force):
        """岛群生成器：产物已存在（同年份）就直接读，否则生成（约 5–15 s）。只读管线产物，不回灌。"""
        ctx = self.app.ctx(rid)
        out = ctx.out_dir / "islands" / str(node)
        if force or not all((out / f).exists() for f in ("island.json", "climate.json", "preview.png", "preview_main.png", f"weather_y{year}.csv")):
            from ..island import generate
            with self.app.island_lock:
                generate(ctx, node, year=year, log=lambda *a: None)
        J = json.loads((out / "island.json").read_text(encoding="utf-8"))
        C = json.loads((out / "climate.json").read_text(encoding="utf-8"))
        return {"node": node, "meta": J["meta"], "constraints": J["constraints"], "layout": J["layout"], "n_islands": len(J["islands"]),
                "islands": [{k: i[k] for k in ("id", "area_km2", "peak_m", "cliff_m", "age_zh", "n_lakes", "has_perennial_river")} for i in J["islands"][:8]],
                "hydro": {k: J["hydro"][k] for k in ("precip_mm", "n_lakes", "lake_km2", "main_basins")}, "landcover": J["landcover"]["share"],
                "climate": {"season_type_zh": C["season_type_zh"], "annual": C["annual"], "thermal": C["thermal"],
                            "seasons": [{k: s[k] for k in ("index", "name", "days", "temp_c", "temp_sea_c", "precip_mm", "storm", "window", "wind", "band_shift_deg")} for s in C["seasons"]]},
                "weather": C.get("weather", {}), "preview": f"/api/island/preview?run={rid}&node={node}&t={int(time.time())}",
                "preview_main": f"/api/island/preview?run={rid}&node={node}&main=1&t={int(time.time())}", "dir": str(out)}

    def _island_data(self, rid, node, year, force=False, sets=None):
        ctx = self.app.ctx(rid)
        out = ctx.out_dir / "islands" / str(node)
        need = force or not all((out / f).exists() for f in ("island.json", "climate.json", "terrain.npz", "preview_main.png", f"weather_y{year}.csv"))
        if not need:
            C = json.loads((out / "climate.json").read_text(encoding="utf-8"))
            need = C.get("weather", {}).get("year") != year or not isinstance(C.get("weather", {}).get("days"), list)
        if need:
            from ..island import generate
            with self.app.island_lock:
                generate(ctx, node, year=year, sets=list(sets or []), log=lambda *a: None)
        J = json.loads((out / "island.json").read_text(encoding="utf-8"))
        C = json.loads((out / "climate.json").read_text(encoding="utf-8"))
        return {"node": node, "run": rid, "year": year, "island": J, "climate": C, "island_cfg": ctx.cfg.get("island", {}),
                "preview": f"/api/island/preview?run={rid}&node={node}&t={int(time.time())}",
                "preview_main": f"/api/island/preview?run={rid}&node={node}&main=1&t={int(time.time())}"}

    def _island_raster(self, rid, node) -> bytes:
        import base64
        import numpy as np
        ctx = self.app.ctx(rid)
        f = ctx.out_dir / "islands" / str(node) / "terrain.npz"
        with np.load(f) as z:
            arrs = {k: z[k] for k in z.files}
        H, W = arrs["height"].shape
        step = 1
        while (H // step) * (W // step) > 1_600_000:
            step += 1
        def pick(a):
            return np.ascontiguousarray(a[::step, ::step])
        h = pick(arrs["height"]).astype(np.float64)
        void = np.isnan(h)
        hq = np.where(void, 0, np.clip(np.round(h), 0, 65534) + 1).astype(np.uint16)   # 0 = 虚空，其余 = 高度 + 1
        out = {"rows": int(hq.shape[0]), "cols": int(hq.shape[1]), "step": step,
               "height_u16": base64.b64encode(hq.tobytes()).decode("ascii"),
               "island_id_i16": base64.b64encode(pick(arrs["island_id"]).astype(np.int16).tobytes()).decode("ascii")}
        for k, dt in (("landcover", np.uint8), ("river", np.uint8), ("stream", np.uint8), ("lake", np.uint8), ("arable", np.uint8), ("cliff", np.uint8)):
            if k in arrs:
                out[k + "_u8"] = base64.b64encode(pick(arrs[k]).astype(dt).tobytes()).decode("ascii")
        if "slope_deg" in arrs:
            out["slope_u8"] = base64.b64encode(np.clip(np.round(pick(arrs["slope_deg"]) * 4), 0, 255).astype(np.uint8).tobytes()).decode("ascii")
        if "flowacc_km2" in arrs:
            fa = pick(arrs["flowacc_km2"]).astype(np.float64)
            out["flowacc_log_u8"] = base64.b64encode(np.clip(np.round(np.log10(np.maximum(fa, 0.0) + 0.01) * 40 + 100), 0, 255).astype(np.uint8).tobytes()).decode("ascii")
        return json.dumps(out).encode("utf-8")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/run":
                rid = self.app.start_run(int(body.get("seed", 42)), list(body.get("sets", [])),
                                         body.get("run_id"))
                self._json({"started": True, "run_id": rid})
            elif self.path == "/api/island/regen":
                sets = [str(x).strip() for x in body.get("sets", []) if str(x).strip()]
                bad = [x for x in sets if not x.startswith("island.") or "=" not in x]
                if bad:
                    self._json({"error": f"参数覆盖只接受 island.a.b=value：{bad}"}, 400)
                else:
                    self._json(self._island_data(body["run"], int(body["node"]), int(body.get("year", 0)), force=True, sets=sets))
            else:
                self._send(b"not found", "text/plain", 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": str(e)}, 400)


def serve(out_root: Path, port: int = 8642, open_browser: bool = True):
    app = App(out_root)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, app=app))
    url = f"http://127.0.0.1:{port}/"
    print(f"操作台：{url}   产物根目录：{Path(out_root).resolve()}   Ctrl+C 退出")
    if open_browser:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def export_static(ctx, out_file: Path, body_only: bool = False) -> Path:
    """单文件导出：同一界面，数据与贴图内嵌，无重跑功能。
    body_only=True 时去掉 doctype/html/head/body 外壳（供嵌入式页面宿主使用）。"""
    import base64
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    if body_only:
        import re
        head = re.search(r"<head>(.*?)</head>", html, re.S).group(1)
        head = re.sub(r"<meta[^>]*>", "", head)
        body = re.search(r"<body>(.*?)</body>", html, re.S).group(1)
        html = head.strip() + "\n" + body
    world = build_world(ctx)
    fields = build_fields(ctx)
    tex = base64.b64encode(build_texture(ctx)).decode("ascii")
    from ..ninegrid import RegionData, build_region_md
    rd = RegionData(ctx)
    grids = {str(r): build_region_md(rd, r)[0] for r in range(rd.n_regions)}
    check_f = ctx.stage_dir(10) / "check.json"
    check = json.loads(check_f.read_text(encoding="utf-8")) if check_f.exists() else {"items": []}
    inline = {"world": world, "fields": fields, "grid": build_grid(ctx),
              "texture": "data:image/png;base64," + tex,
              "ninegrid": grids, "check": check, "config": ctx.cfg}
    payload = "<script>window.__INLINE__=" + dumps(inline).replace("</", "<\\/") + ";</script>"
    html = html.replace("<!--INLINE-->", payload)
    # 内嵌 globe.gl：单文件完全离线
    vendor = (STATIC / "vendor" / "globe.gl.min.js").read_text(encoding="utf-8")
    html = html.replace('<script src="vendor/globe.gl.min.js"></script><!--VENDOR-->',
                        "<script>" + vendor.replace("</script", "<\\/script") + "</script>")
    out_file.write_text(html, encoding="utf-8")
    return out_file
