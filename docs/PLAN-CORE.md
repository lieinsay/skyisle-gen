# 生成器后端换 C++：核心库设计稿（行星计划 P6a–d，skyisle-gen 侧）

上游：Zhouzhu 的 `docs/PLAN-PLANET.md`（D1 生成器是唯一来源、D3 后端整体换 C++、D15 两种玩法两种世界、D16 仓库不合并算法合一、4.9 分工、第五节 P6）。
本稿管 skyisle-gen 这一侧：核心库 `core/` 怎么划、按什么顺序移、Python 前端怎么调它、怎么对照验收。**P6a 只移第三层的地形段**（布局含势力范围、岛形、地形、水系、河道成形），
其余（资源、聚落、气候、天气、粗版的降采样、行星层 ①–⑨）照旧走 Python，P6b–d 接着移。

## 一、范围与不变的东西

- 核心库 `core/`：C++17，**不含 Python、不含 Godot**，CMake 构建；Zhouzhu 以后以子模块只编它（D16）。
- Python 前端（命令、缓存与产物 IO、check、九格表、出图、操作台）不变；经 nanobind 绑定调核心库，**用法与产物格式（npz / json / png）不变**。
- 每移完一段，前端加开关 `[engine] backend = "python" | "cpp"`，两边都能跑；统计对照与校验都过了再往下。这一期**不切默认**（仍是 python）。
- **python 后端的产物一字不改**（2051 逐字节对照）；cpp 后端只在 `island.json` 的 `meta` 多一个 `"engine": "cpp"`。
- 第三层的规矩照旧：不回灌、`stages/` 不 import `island`、随机数只从 `entity_rng(seed, 21, "island:{node}:{部件}")` 取。

## 二、目录与模块

```
core/
  CMakeLists.txt            静态库 skyisle_core；可选：Python 扩展 _core（找得到 nanobind 时）、自检程序 core_selftest（ctest）
  build.py                  一键构建（Windows 自动进 VS 2022 的 x64 环境；Linux 直接 cmake）；.pyd / .so 落到 skyisle_gen/
  include/skyisle/
    rng.hpp                 crc32、SeedSequence、PCG64（XSL-RR）、Rng（random / uniform / integers / normal / gamma / beta / choice）、entity_rng
    grid.hpp                Grid<T>（行主序）、N4 / N8、LatticeNoise / FractalNoise、连通分量、形态学、distance_bands、nearest_propagate、
                            block_mean / block_any / upsample_bilinear / smooth121 / laplacian / slope_deg、numpy 同式的成对求和与中位数、Python 同式的 round
    flow.hpp                priority_fill、fill_iter、d8、d8_random、accumulate、按下游排序（拓扑序）
    config.hpp              Config：扁平键值表（"terrain.carve_k" → double，另有 double 列表），缺键抛异常
    island/
      types.hpp             NodeInputs（本群的行星层标量）、PlanetView（板块网格、全体群心与陆地、局地风网格）、Group（群栅格与各岛记录）
      layout.hpp            5.1：岛数、Zipf、boundary_axis、radial_profile、place_islands、台面、起伏、岸距、索桥 / 短渡 / 导水槽
      territory.hpp         势力范围：limits、mask_support、profile_support、violation、nearest_fit、raster_violation
      terrain.hpp           5.2：island_shape、base_form、erode、fit_rim、sculpt_island
      river.hpp             5.3b：carve_channels、trace_lines
      hydro.hpp             5.3：build_hydro（湖、路由面、D8、汇流、河 / 溪、盆地、河道成形、地表 12 类、可耕地）
      build.hpp             build_terrain（布局 → 岛形 → 势力范围 → 贴栅格 → 岸距与连接 → 崖缘）
  src/                      同名 .cpp；ziggurat_tables.inc（numpy 的 ziggurat 表，探出来的，见四）
  python/module.cpp         nanobind 绑定（_core）
  tests/selftest.cpp        C++ 自检（随机数对 numpy 的参考值、确定性）
skyisle_gen/island/engine.py   后端开关与 cpp 桥：把 [island] 段展平成键值表、准备 NodeInputs / PlanetView、调 _core、把结果拼回和 Python 版同形的 g
skyisle_gen/island/compare.py  `skyisle island compare`：分层抽样 N 群，两个后端各跑一遍，统计对照 + island check + 用时
```

C++ 里的字符串一律 ASCII（枚举与键名），中文名（新岛 / 中年 / 老岛、地表类名）由 Python 前端映射；注释可以写中文（MSVC 加 `/utf-8`）。

## 三、移植顺序（这一期）

每一步都有「同输入对照」：绑定把公共件单独暴露出来，pytest 拿同一份输入比 C++ 与 numpy 的输出（能逐位的逐位比，不能的比容差）。

1. **随机数**：SeedSequence + PCG64 + 各分布。对照：原始 64 位输出、uniform / integers / normal / beta / choice 与 numpy 逐位相同。
2. **栅格公共件**：值噪声 / 分形噪声（给同一流 → 逐位同）、连通分量（编号顺序同 numpy 版：按光栅扫描首次出现）、腐蚀 / 膨胀、distance_bands、nearest_propagate、
   块均值、双线性放大、平滑、拉普拉斯、坡度。对照：逐位或 1e-12。
3. **水文核心**：priority_fill（堆的次序与 heapq 同：(z, i, j)）、fill_iter（Jacobi，只算邻居变了的格，结果同全量迭代）、d8（同向序、同严格大于）、
   d8_random、accumulate（拓扑序累加；计数是整数，与按高度排序逐位同）。对照：逐位。
4. **布局与势力范围**：island_count / zipf_sizes / surface_heights / relief_targets / boundary_axis / radial_profile / place_islands / territory.* / shoreline_gaps / links。
5. **岛形与地形**：island_shape（面积二分）、base_form（三种岛龄）、erode（隐式下切 + 坍塌 + 扩散）、sculpt_island、build_terrain 的贴栅格与裁切。
6. **水系与河道**：build_hydro（每岛切片：填洼、湖、抬洼、路由面、D8、汇流、河阈、盆地、carve_channels、trace_lines）、台面校正、地表分类、可耕地。
7. **前端开关**：`build_terrain` / `build_hydro` 的入口按后端分派（`generate`、`island lod`、操作台重生成、`check`、`batch` 都经过这两处），产物照旧由 Python 写。
8. **对照工具与用时**：`skyisle island compare`。

## 四、随机数（比计划多做的一步）

计划说随机数不必与 numpy 逐位一致（D3）。实际做下来 numpy 这几样的算法都是公开的定式，追得上，而且追上了好处很大：
两个后端大多数群的布局、岛形完全一样，对照从「统计上像」变成「逐群可比」，移植错误一眼看得出。所以这一期**把随机流做成与 numpy 逐位一致**：

- `SeedSequence`（numpy 的 hashmix / mix，池 4 字）→ `PCG64`（128 位 LCG + XSL-RR 输出，先步进后输出）；`entity_rng(seed, stream, key)` = `SeedSequence([seed, stream, crc32(key)])`。
- `random()` = (next_u64 >> 11) × 2⁻⁵³；`uniform(lo, hi)` = lo + (hi − lo) × random()；`integers(lo, hi)` = 32 位 Lemire（缓存半个 64 位字，与 numpy 的 `has_uint32` 同）；
- `standard_normal` = numpy 的 256 层 ziggurat。表不在 numpy 的发行包里：`wi`、`ki` 用「设 PCG64 状态让下一个原始输出等于指定值」逐层探出来（wi 取 rabs = 1 时的输出、ki 二分找快慢路径的分界），
  `fi = exp(−x²/2)`（x = wi × 2⁵²）。探表脚本与表一起提交（`core/tools/probe_ziggurat.py`）。
- `standard_gamma`（Marsaglia–Tsang，形状 > 1）、`beta`（两伽马之比；a、b ≤ 1 时 Jöhnk）、`choice(k, p)`（累积和 + searchsorted 右侧）与 numpy 的 C 源同式。

**不保证**两边结果逐位一致：exp / log / pow / atan2 在 numpy（SIMD）与 C 库之间可能差最后一位，numpy 的求和是成对求和（前端里用到的地方 C++ 照做了），
差一位就可能让某个阈值判断翻过去、之后整个群分叉（例如 place 流在某次尝试上接受与否不同）。所以验收仍按统计对照，逐群一致只是常态、不是约束。（实测：P6a 抽到的 270 群地形 + 水系全部逐位相同，见 DESIGN-NOTES 四点二十三。）

## 五、绑定接口（`skyisle_gen._core`）

两层：**步**（前端正式调用）与**公共件**（给 pytest 做同输入对照、给调试用）。数组一律 numpy、C 连续、行主序；传入时复制一份，返回的数组由 C++ 分配、numpy 持有。

| 函数 | 输入 | 输出 |
|---|---|---|
| `build_terrain(inp, planet, cfg, res_m=None, threads=0)` | inp：本群标量（节点号、seed、lat / lon、陆地、主岛、台面、岛龄、叠层、keel 余隙）；planet：板块网格（lat0 / dlat / lon0 / dlon + K + btype + lats / lons）、全体群的 lat / lon / 陆地、行星半径；cfg：展平的 [island] | dict：H / W / res_km / x0 / y0、height（f64，虚空 NaN）/ island_id（i16）/ cliff（bool）、各岛记录（面积格数、目标、岛心、台面、起伏目标、岸缘、峰、岛底、岛龄、类别、外框）、links、导水槽树、势力范围记录、masks_pos、轴向 / 边界核 / 类型、用时 |
| `build_hydro(state, planet, cfg, threads=0)` | state：build_terrain 的数组与各岛的 rim / keel / age（**用 island.json 里已四舍五入的值**，与 Python 版同口径）、origin_km（同样是四舍五入后的）；planet：局地风网格、年长秒数；inp 的降水 / 海面温 / 直减率 / 有河 / 河级 / 可耕率 | dict：height（改后）、filled、flowacc、river / stream / lake、宽 / 深 / 漫滩 / 下切、recv_i / recv_j / route_h、landcover、arable、slope、各岛水系摘要、主岛盆地、河口表、河道折线、台面校正量 dz |
| 公共件 | `rng_raw / rng_draw / rng_choice_p`（各分布）、`crc32`、`np_sum`、`pyround`、`math_fns`（numpy 同式的幂与斜边）、`grid_interp`、`fractal_noise`、`label_components`、`largest_component`、`binary_erode / dilate`、`distance_bands`、`nearest_propagate`、`block_mean / any`、`upsample_bilinear`、`smooth121`、`laplacian`、`slope_deg`、`priority_fill`、`fill_iter`、`d8`、`d8_random`、`accumulate`、`island_shape`、`radial_profile`、`sculpt_island`、`territory_limits`、`nearest_fit`、`boundary_axis` | 同 Python 版的返回 |

配置用**扁平键值表**：前端把 `[island]` 段展平成 `{"layout.n0": 30.0, "territory.stretch": [1.0, 1.6, 2.4], "territory.enabled": 1.0, …}`（布尔 → 0 / 1，字符串跳过），
C++ 按键取、缺键抛异常。以后游戏从 TOML / 行星包读同一张表。`[engine] threads`（默认 4）控制群内各岛并行的线程数，结果与线程数无关（每岛自己的随机流、写各自的格）。

**J（island.json）的拼装留在前端**：C++ 给原始的双精度数，前端照 Python 版的键序与 `round()` 位数拼（`engine.py`），所以两个后端的 island.json 同形。
C++ 内部凡是 Python 版读的是「已四舍五入的 J 值」的地方（hydro 用的 origin_km、rim_m、keel_m，导水槽 Prim 堆里的 gap_km），C++ 也按 Python 的 `round` 取（`pyround`：printf 正确舍入 + strtod，与 CPython 的 round 同为「二进制值的精确十进制舍入、逢半取偶」）。

## 六、前端开关

- `config/default.toml` 新增 `[engine]`：`backend = "python"`、`threads = 4`。读法与 `[island]` 同：default.toml ← run 的 config.resolved.toml ← `--set engine.backend=cpp`；
  `skyisle island … --backend cpp` 是它的简写。**不进任何阶段的缓存 key**（第三层本来就不进；P6c 起行星层的阶段要把它并进 key）。
- 分派点只有两处：`island.build_terrain` 与 `hydro.build_hydro` 的第一行（`generate`、`lod`、操作台、check、batch 都经过它们）。cpp 后端但扩展没编：报错并给出构建命令（不静默退回，免得以为跑的是 C++）。
- cpp 后端下后面的步（资源、气候、天气、聚落、出图、写产物）照旧是 Python，读的是同形的 g。

## 七、构建

```
$py -m pip install nanobind                       # 3.x；装不上就换 pybind11（绑定只有一个文件）
$py core/build.py                                  # Release；--debug 调试版；--test 另跑 ctest 自检；--clean 重来
```
Windows：`build.py` 用 vswhere（或 `D:\Program Files\Microsoft Visual Studio\2022\Community`）找 `vcvars64.bat`，在那个环境里跑 CMake + Ninja；
Linux（ME Pro）：直接 `cmake -G Ninja`（没有 Ninja 用 Makefiles）。产物 `skyisle_gen/_core.cp312-win_amd64.pyd`（Linux `.so`）已 gitignore。
编译选项：MSVC `/O2 /utf-8 /EHsc /permissive-`（默认 `/fp:precise`、不开 AVX2，免得乘加被合并成 FMA 与 numpy 差位）；GCC / Clang `-O2 -ffp-contract=off`。
`pip install -e .` 不变（setuptools）；可编辑安装指向源码目录，编好的扩展就在包里。

## 八、对照与验收（这一期）

`skyisle island compare --run out/seed42 --sample 30 [--jobs N]`：用 `island batch` 的分层抽样（主岛面积四分位），每群两个后端各跑完整的 `generate` + `island check`
（前 3 群加 IS-det 重跑），产物写 `islands_compare/<后端>/<节点>/`（不覆盖 `islands/`），汇总写 `islands/compare.json`：

- 陆地、主岛：两个后端都按目标精确（IS-area < 2%）；岛数相同；
- 峰高 ±10%；河长（主岛常年河中心线总长）±20%；可耕率（都 = 目标 ± 0.005）、地表 12 类占比、湖数、河口数也报出来；
- 两边的 island check 全过；
- 用时：每群地形、水系分开记（进程里计时）；干净的数另用 `island compare --timing` 顺序跑（python、cpp 1 线程、cpp 4 线程各一遍，先热身）。

另：pytest 加 `tests/test_core_engine.py`（扩展没编就跳过）——随机数逐位、公共件同输入对照、小世界上两个后端的 build_terrain / build_hydro 对照、cpp 后端 IS-det、线程数不影响结果。

## 九、以后各期从哪接

- **P6b**（第三层其余）：resources（赋存场、点片散、采场）、settle / tiers、climate、weather、lod 的降采样挪进 `core/island/`，`generate` 整个进 C++（产物写出仍在前端）。
  这一期留下的 `Group` 结构就是 g 的 C++ 形；`build_hydro` 返回的 recv / route_h / cut_m 是资源层要的。
- **P6c / P6d**（行星层）：`core/planet/` 按阶段移，`[engine] backend` 并进阶段缓存 key；三 seed 的 check 在 C++ 后端下全过后切默认、删 Python 算法。
- 游戏（P6e）：`core/` 作子模块编进 GDExtension；`build_terrain` / `build_hydro` 已经不依赖 Python，NodeInputs / PlanetView 由行星包填。
