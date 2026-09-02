"""集成测试：确定性（同 seed 同世界）与缓存链（docs/12 §七）。小规模，较慢。"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zhouzhu_gen.config import load_config, apply_sets
from zhouzhu_gen.pipeline import run

SMALL = ["s03.islands.n_islands=800", "s07.regions.n_regions=12",
         "s07.regions.max_regions=20", "s06.routes.betweenness_sources=48",
         "s08.k_sub=1"]


def _hash_dir(d: Path) -> dict:
    out = {}
    for p in sorted(d.rglob("*.npz")):
        out[p.relative_to(d).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("det")
    cfg = load_config(sets=SMALL + ["run.id=a"])
    out_a = run(cfg, 7, root, upto=8)
    cfg2 = load_config(sets=SMALL + ["run.id=b"])
    out_b = run(cfg2, 7, root, upto=8)
    return out_a, out_b


def test_determinism_bytewise(two_runs):
    a, b = two_runs
    ha, hb = _hash_dir(a), _hash_dir(b)
    assert ha == hb, "同 seed 双跑产物不一致"


def test_stage8_param_change_keeps_upstream(two_runs, tmp_path):
    a, _ = two_runs
    before = _hash_dir(a)
    cfg = load_config(sets=SMALL + ["run.id=a", "s08.eps0=0.05"])
    run(cfg, 7, a.parent, upto=8)
    after = _hash_dir(a)
    for k in before:
        if k.startswith(("s01", "s02", "s03", "s04", "s05", "s06", "s07")):
            assert before[k] == after[k], f"只改 s08 参数却重算了 {k}"
    assert before["s08_diffusion/iso.npz"] != after["s08_diffusion/iso.npz"] or \
        before["s08_diffusion/fields.npz"] != after["s08_diffusion/fields.npz"]


def test_prehist_covers_everyone(two_runs):
    """原则己：史前扩散全覆盖。"""
    import numpy as np
    a, _ = two_runs
    with np.load(a / "s07_centers" / "prehist.npz") as z:
        assert np.isfinite(z["dist_pre"]).all()


def test_share_normalized(two_runs):
    import numpy as np
    a, _ = two_runs
    meta = json.loads((a / "s08_diffusion" / "traits.resolved.json").read_text(encoding="utf-8"))
    with np.load(a / "s08_diffusion" / "fields.npz") as z:
        share = z["share"]
    with np.load(a / "s08_diffusion" / "iso.npz") as z:
        local = z["local_share"]
    slot_of = {}
    for t in meta["traits"]:
        slot_of.setdefault(t["slot"], []).append(t["index"])
    for si, (slot, rows) in enumerate(sorted(slot_of.items())):
        tot = share[rows].sum(axis=0) + local[si]
        assert np.allclose(tot, 1.0, atol=2e-3), f"槽位 {slot} 未归一"
