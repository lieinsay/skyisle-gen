"""测试的公共设置。

P6d 起默认后端是 cpp（[engine] backend = "cpp"）。C++ 扩展没编的机器上（如还没编过的 ME Pro），不依赖 C++ 的测试照旧用
python 参考后端跑：这里把 load_config 包一层，先塞一条 engine.backend=python（测试自己给的 --set 在后，照样覆盖）；
依赖 C++ 的测试文件自己 importorskip("skyisle_gen._core")，没编就整个跳过。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import skyisle_gen._core  # noqa: F401
except ImportError:
    import skyisle_gen.config as _C

    _orig = _C.load_config

    def _load_config_python(paths=None, sets=None):
        return _orig(paths, ["engine.backend=python"] + list(sets or []))

    _C.load_config = _load_config_python
