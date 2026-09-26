"""一键构建 C++ 核心库与 Python 扩展（docs/PLAN-CORE.md 第七节）。

    python core/build.py            # Release（默认），扩展落到 skyisle_gen/_core.*.pyd / .so
    python core/build.py --debug    # Debug（另一个构建目录 build/core-debug）
    python core/build.py --test     # 编完跑 ctest（C++ 自检）
    python core/build.py --clean    # 删掉构建目录重来

Windows：cl.exe 不在 PATH 时，用 vswhere（或 D:\\Program Files\\Microsoft Visual Studio\\2022\\Community）找 vcvars64.bat，在那个环境里跑 CMake + Ninja。
Linux：直接 cmake；有 ninja 用 Ninja，没有用 Unix Makefiles。
需要：CMake ≥ 3.20、C++17 编译器、`python -m pip install nanobind`（装在跑这个脚本的 Python 里）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "core"
VS_FALLBACK = [r"D:\Program Files\Microsoft Visual Studio\2022\Community",
               r"C:\Program Files\Microsoft Visual Studio\2022\Community",
               r"C:\Program Files\Microsoft Visual Studio\2022\Professional",
               r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools"]


def _vcvars() -> Path | None:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    cands = []
    if vswhere.exists():
        r = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                            "-property", "installationPath"], capture_output=True, text=True)
        cands += [x.strip() for x in r.stdout.splitlines() if x.strip()]
    cands += VS_FALLBACK
    for c in cands:
        p = Path(c) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if p.exists():
            return p
    return None


def _msvc_env() -> dict:
    """在 vcvars64.bat 的环境里取出环境变量（cl.exe 已在 PATH 就用当前环境）。"""
    env = dict(os.environ)
    if shutil.which("cl"):
        return env
    bat = _vcvars()
    if bat is None:
        sys.exit("找不到 Visual Studio 2022 的 vcvars64.bat（装「使用 C++ 的桌面开发」工作负载，或在「x64 Native Tools」命令行里运行本脚本）")
    # cmd 的 set 按控制台代码页（中文 Windows 是 cp936）输出；PYTHONUTF8=1 时 text=True 会按 UTF-8 解码失败，环境变量整段丢掉
    out = subprocess.run(f'cmd /s /c ""{bat}" >nul && set"', capture_output=True, text=True, shell=True, encoding="oem", errors="replace")
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 skyisle-gen 的 C++ 核心库与 Python 扩展")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--test", action="store_true", help="编完跑 ctest")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--jobs", type=int, default=0)
    a = ap.parse_args()
    cfg = "Debug" if a.debug else "Release"
    bdir = ROOT / "build" / ("core-debug" if a.debug else "core")
    if a.clean and bdir.exists():
        shutil.rmtree(bdir)
    env = _msvc_env() if os.name == "nt" else dict(os.environ)
    if os.name == "nt":
        env.setdefault("VSLANG", "1033")   # 编译器报错用英文（中文版 VS 的报错是 GBK，经管道读出来是乱码）
    gen = "Ninja" if shutil.which("ninja", path=env.get("PATH")) else ("NMake Makefiles" if os.name == "nt" else "Unix Makefiles")
    try:
        import nanobind
        nb_dir = nanobind.cmake_dir()
    except ImportError:
        sys.exit(f"没装 nanobind：{sys.executable} -m pip install nanobind")
    conf = ["cmake", "-S", str(CORE), "-B", str(bdir), "-G", gen, f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPython_EXECUTABLE={sys.executable}", f"-Dnanobind_DIR={nb_dir}"]
    if not (bdir / "CMakeCache.txt").exists():
        print(" ".join(conf))
        subprocess.run(conf, env=env, check=True)
    build = ["cmake", "--build", str(bdir), "--config", cfg]
    if a.jobs:
        build += ["--parallel", str(a.jobs)]
    print(" ".join(build))
    subprocess.run(build, env=env, check=True)
    if a.test:
        subprocess.run(["ctest", "--test-dir", str(bdir), "-C", cfg, "--output-on-failure"], env=env, check=True)
    built = sorted((ROOT / "skyisle_gen").glob("_core*"))
    print("扩展：" + (", ".join(p.name for p in built) if built else "（没有生成）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
