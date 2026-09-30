"""测试的公共设置。

算法只在 C++ 核心里（Python 参考后端 2026-09-30 删了，git tag python-reference-final）。依赖 C++ 的测试文件自己
importorskip("skyisle_gen._core")，扩展没编（python core/build.py）就整个跳过；test_core.py 里读源码的静态断言与配置校验没编也照跑。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
