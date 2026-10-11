"""同一次原生生成在资源/聚落之前取多档 LOD，保留细包和天气。"""
import numpy as np
import pytest

from test_core_p6b import small_ctx, _nodes  # 复用已存在的小世界 fixture。


def test_native_hydro_lod_capture(small_ctx, monkeypatch):
    from skyisle_gen import island as isl
    from skyisle_gen.island import engine
    from skyisle_gen.island.hydro import build_hydro
    from skyisle_gen.island.lod import weather_arrays

    node = _nodes(small_ctx, 1)[0]
    args = dict(res_m=300.0, steps=5, write=False, log=lambda *a: None)
    original = isl.generate(small_ctx, node, **args)
    native_core = engine.core()
    calls = []

    class CountCore:
        def __getattr__(self, name):
            method = getattr(native_core, name)
            if name not in ("generate", "build_terrain", "build_hydro"):
                return method

            def call(*a, **kw):
                calls.append(name)
                return method(*a, **kw)
            return call

    with monkeypatch.context() as patch:
        patch.setattr(engine, "core", lambda: CountCore())
        captured = isl.generate(small_ctx, node, lod_resolutions=[1800.0, 900.0, 600.0, 900.0], **args)
    assert calls == ["generate"]
    assert list(captured["lods"]) == [1800.0, 900.0, 600.0]
    for key in ("height", "island_id", "landcover", "recv_i", "recv_j", "w_ch_m", "d_ch_m"):
        assert np.array_equal(original[key], captured[key], equal_nan=True), key
    c = isl.island_config(small_ctx, [])
    a, b = weather_arrays(original, c), weather_arrays(captured, c)
    for key in a:
        assert np.array_equal(a[key], b[key]), key

    inp = isl._node_inputs(small_ctx, node)
    native = isl.build_terrain(small_ctx, node, c, inp, res_m=300.0, log=lambda *a: None)
    build_hydro(small_ctx, node, c, native, log=lambda *a: None)
    for res, block in captured["lods"].items():
        expected = engine.block_reduce_cpp(native, int(res / 300.0))
        for key in expected:
            assert np.array_equal(expected[key], block[key], equal_nan=True), (res, key)


def test_lod_capture_requires_hydro(small_ctx):
    from skyisle_gen import island as isl
    with pytest.raises(ValueError, match="hydro stage"):
        isl.generate(small_ctx, _nodes(small_ctx, 1)[0], steps=1, write=False, lod_resolutions=[2000.0], log=lambda *a: None)
