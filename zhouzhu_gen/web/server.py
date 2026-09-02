"""本地操作台服务（仅标准库 http.server）。

GET  /                        操作台页面
GET  /api/runs                产物目录列表
GET  /api/world?run=          世界数据包
GET  /api/fields?run=         特征场（量化）
GET  /api/texture?run=        行星贴图 PNG
GET  /api/config?run=         生效配置
GET  /api/check?run=          验收报告
GET  /api/ninegrid?run=&region=   九格表 markdown
GET  /api/path?run=&a=&b=&mode=   最优路径逐跳
POST /api/run  {seed, sets:[...], base_run}   后台重跑管线
GET  /api/run/status          进度
"""
from __future__ import annotations

import json
import threading
import traceback
import urllib.parse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..cli import _ctx_from_run
from .bundle import build_fields, build_texture, build_world, dumps

STATIC = Path(__file__).parent / "static"


class App:
    def __init__(self, out_root: Path):
        self.out_root = Path(out_root)
        self.cache: dict[tuple, object] = {}
        self.lock = threading.Lock()
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
                pipeline_run(cfg, seed, self.out_root, upto=9, log=self.job["log"].append)
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
            elif p == "/api/texture":
                rid = q["run"]
                body = self.app.cached(("texture", rid), lambda: build_texture(self.app.ctx(rid)))
                self._send(body, "image/png")
            elif p == "/api/config":
                rid = q["run"]
                self._json(self.app.ctx(rid).cfg)
            elif p == "/api/check":
                rid = q["run"]
                f = self.app.out_root / rid / "s09_output" / "check.json"
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

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/run":
                rid = self.app.start_run(int(body.get("seed", 42)), list(body.get("sets", [])),
                                         body.get("run_id"))
                self._json({"started": True, "run_id": rid})
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
    check_f = ctx.stage_dir(9) / "check.json"
    check = json.loads(check_f.read_text(encoding="utf-8")) if check_f.exists() else {"items": []}
    inline = {"world": world, "fields": fields, "texture": "data:image/png;base64," + tex,
              "ninegrid": grids, "check": check, "config": ctx.cfg}
    payload = "<script>window.__INLINE__=" + dumps(inline).replace("</", "<\\/") + ";</script>"
    html = html.replace("<!--INLINE-->", payload)
    # 内嵌 globe.gl：单文件完全离线
    vendor = (STATIC / "vendor" / "globe.gl.min.js").read_text(encoding="utf-8")
    html = html.replace('<script src="vendor/globe.gl.min.js"></script><!--VENDOR-->',
                        "<script>" + vendor.replace("</script", "<\\/script") + "</script>")
    out_file.write_text(html, encoding="utf-8")
    return out_file
