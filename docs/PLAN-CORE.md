# 生成器后端换 C++：核心库设计稿（行星计划 P6a–d，skyisle-gen 侧；P6a–d 已完成）

上游：Zhouzhu 的 `docs/PLAN-PLANET.md`（D1 生成器是唯一来源、D3 后端整体换 C++、D15 两种玩法两种世界、D16 仓库不合并算法合一、4.9 分工、第五节 P6）。
本稿管 skyisle-gen 这一侧：核心库 `core/` 怎么划、按什么顺序移、Python 前端怎么调它、怎么对照验收。**P6a 移了第三层的地形段**（布局含势力范围、岛形、地形、水系、河道成形），
**P6b 移了第三层其余**（资源、聚落与层级、四季、逐日天气、粗版的降采样，`generate` 整个进 C++，写产物仍在前端）；
**P6c 移了行星层 ①–④**（行星与历法、风带、岛群分布与板块与候选边、局地风与气候 / 水汽模型；`core/planet/`，npz / json 仍由前端写）；
**P6d 移了行星层 ⑤–⑨**（障碍、航路与抽样介数、文明中心与地区、特征扩散、政治层）并**把默认后端切到 C++**；Python 版的算法冻结成参考后端（只作对照），
**Linux（ME Pro）上编过、验过逐位相同之后 2026-09-30 删了**（DESIGN-NOTES 四点四十一 / 四点四十二；git tag `python-reference-final` 是删之前的最后一版）。
⑩ 输出（九格表、出图）只在 Python。下文讲的 `[engine] backend` 开关、`island compare`、两个后端的对照都是移植期间的做法，现在只剩 C++ 一份；
C++ 注释里「xxx.py 同式」指 tag 里的那一版 Python。

## 一、范围与不变的东西

- 核心库 `core/`：C++17，**不含 Python、不含 Godot**，CMake 构建；Zhouzhu 以后以子模块只编它（D16）。
- Python 前端（命令、缓存与产物 IO、check、九格表、出图、操作台）不变；经 nanobind 绑定调核心库，**用法与产物格式（npz / json / png）不变**。
- 每移完一段，前端加开关 `[engine] backend = "python" | "cpp"`，两边都能跑；统计对照与校验都过了再往下。P6a–c 不切默认；**P6d 切到 cpp**
  （三 seed 的 check 在 cpp 下全过、①–⑨ 与 python 逐位相同之后）。python 冻结成参考后端：新改动先在 C++ 里做，python 只作对照。
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
      resources.hpp         5.3c（P6b）：地形区、点 / 片、赋存场 → 赋存区 → 采场、sync_resources、rock_site_mask、open_working
      climate.hpp           5.4 / 5.5（P6b）：build_climate、daily_curves、season_params、simulate_year、multi_year（IS-daily）
      settle.hpp            聚落与层级（P6b）：build_settlements（settle.py + tiers.py）、kmeans_split
      generate.hpp          整群 generate（steps 1–5 同 Python 版）与粗版块降采样 block_reduce
    planet/                 （P6c）行星层 ①–④；（P6d）⑤–⑨
      planet.hpp            产物结构（Planet / Winds / Islands / Climate，同 npz 的字段，浮点存双精度原值）、stage1–4、run（一次跑完）、
                            球面与经纬网格公共件（grid_axes、latlon_to_xyz、angdist、knn、fractal_noise、gradient / divergence、np_std、rfft_lowpass）、
                            wind_profile / g_vortex、plate_fields、season_range / insolation_first_harmonic、np_maximum / np_minimum
      view.hpp              行星层 → 第三层：planet_view（PlanetView）、node_inputs（NodeInputs）、island_calendar、median_f32
      graph.hpp             （P6d）graph.py 的 C++ 版：CSR（按 (src, dst) 稳定排序）、dijkstra（多源 / 有界，堆次序同 heapq 的 (dist, node)）、
                            accumulate_along_tree、betweenness_sampled（按源分块并行、按源的次序归约）、weak_components、argsort_stable
      civ.hpp               （P6d）⑤–⑨ 的产物结构（Barriers / Routes / Centers / Diffusion / Polity，同 npz 的字段；json 的记录要的原始数）、
                            stage5–9、directed / lambda_ref / mode_weight（weights.py）、population、apply_polity（⑨ → NodeInputs）、run_society（①–⑨ 一次跑完）
    json.hpp                极简 JSON 值（有序对象、整数与浮点分开）：资源 / 聚落 / 四季的记录交给前端与游戏的形（字符串是 ASCII 代码）
  src/                      同名 .cpp；ziggurat_tables.inc（numpy 的 ziggurat 表，探出来的，见四）；planet/（sphere、stage12、stage3、stage4、view；P6d：graph、stage5–9）
  third_party/pocketfft/    pocketfft_hdronly.h（BSD 3-Clause；numpy 2.5.2 引用的同一提交 33ae5dc）：④ 带界位移的 np.fft.rfft / irfft
  python/module.cpp         nanobind 绑定（_core）；bind_island.cpp（第三层）、bind_planet.cpp（P6c 行星层 ①–④）、bind_civ.cpp（P6d ⑤–⑨）；bind_util.hpp 数组转换
  tests/selftest.cpp        C++ 自检（随机数对 numpy 的参考值、确定性）
skyisle_gen/engine.py          （P6c / P6d）后端开关（backend / core；默认 cpp）、行星层配置展平（planet_config：shared / skeleton / s01–s09 / slots / traits_manual）、
                               阶段 key 的后端分量、各步 C++ 对象的进程内缓存与从 npz / json 读回（part；⑧ 不读回）
skyisle_gen/island/engine.py   第三层的 cpp 桥：把 [island] 段展平成键值表、取 PlanetView / NodeInputs（P6c 起由 C++ 从行星层对象给）、调 _core、把结果拼回和 Python 版同形的 g
skyisle_gen/island/decode.py   （P6b）C++ 记录里的 ASCII 代码 → 中文、带数的备注按 Python 版的 f-string 拼；赋存区的长度 / 走向按 numpy 重算
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

P6b（第三层其余）接着：

9. **公共件补齐**：指数 ziggurat（伽马形状 < 1 用）、泊松（乘积法 / PTRS）、对数正态、不放回抽签（Floyd + 洗牌 / 尾部洗牌）；label_by_island、window_extrema、
   np.quantile（linear）、np.interp、np.convolve（核长 < 12 顺序乘加，≥ 12 走 BLAS ddot）、float32 成对求和；Python 内置 sum() 的 Neumaier 补偿求和、numpy 标量的 round。
10. **资源**（resources.py）→ **四季 / 逐日曲线 / 天气**（climate.py、weather.py）→ **聚落与层级**（settle.py、tiers.py）→ **粗版块降采样**（lod._block_reduce）→ **整群 generate**。
11. **前端**：`island.generate` 第一处分派（cpp 后端一次调 `_core.generate`）；资源 / 四季 / 天气 / 聚落的拼装与 island.json 摘要抽成两个后端共用的函数
    （resource_record / resource_summary / set_climate / set_weather / set_settlements），python 后端产物逐字节不变。
12. **对照工具加判据**：村数、户数、资源处数、雨日比例、季型，外加整套产物逐字节对照；`--timing` 改量整群 generate。

P6c（行星层 ①–④）接着：

13. **公共件**：经纬网格（grid_axes）、球面几何（xyz、角距、叉积）、kNN（BLAS dgemm 的乘加次序）、经度周期的分形值噪声（noise.py）、
    np.gradient、球面散度、ndarray.std、np.fft.rfft / irfft 的低通（第三方 pocketfft，与 numpy 同一份）、复数的成对求和与 np.abs、np.maximum / minimum 的平局。
14. **① 行星与历法**（s01_planet、almanac）→ **② 风带与 G**（s02_wind）→ **③ 岛群分布**（s03_islands、tectonics、graph.weak_components）
    → **④ 局地风与气候**（s04_climate、localwind、moisture、skeleton.season_range）→ **run**（一次跑完）。
15. **前端**：s01–s04 的 run() 第一行按后端分派，产物与摘要的写出抽成两个后端共用的 `_write`（python 后端产物逐字节不变）；
    `[engine] backend` 并进 ①–④ 的阶段缓存 key（cpp 另混入 "+cpp"，python 不变）；`run / stage --backend`。
16. **第三层接行星层**：`planet_view / node_inputs` 从 ①③④ 的产物对象直接给（同进程刚跑过 cpp 的 ①–④ 就用内存里的，不经 npz）。

P6d（行星层 ⑤–⑨ 与切换）接着：

17. **公共件**：graph.hpp（CSR、Dijkstra 的 heapq 平局次序与 inf 边、有界搜索、沿树累加、Brandes 抽样介数、弱连通分量）；
    Rng::choice_noreplace_p（Generator.choice(n, size, replace=False, p) 的逐轮去重）；weights（λ_ref、L = −ln perm、模式权重）。
18. **⑤ 障碍**（s05_barriers：Φ 穿越归一化的四种 kind、边局部因子、政治性障碍、G 的大圆弧段阻断）→ **⑥ 航路**（s06_routes：沿边 slerp 取样、有向与分模式成本、
    抽源与介数、枢纽分位、每模式连通分量）→ **⑦ 文明中心**（s07_centers：适宜度与平滑、窗内涌现与放宽、次级极大与每圈保底、史前扩散与谱系、地区、干线）
    → **⑧ 特征扩散**（s08_diffusion：隔离度、特征表（含手工特征表）、reach 与 fast 参考树、对称不动点、conflict_by）→ **⑨ 政治层**（s09_polity 全部）→ **run_society**。
19. **前端**：s05–s09 的 run() 第一行分派、`_write` 两边共用（中文名、round 的位数、numpy 标量的 round 都在前端）；engine.CPP_STAGES = 1–9；行星层配置多展平 s05–s09、槽位表、手工特征表。
20. **第三层**：NodeInputs 的人口与邦都由 C++ 从 ⑨ 的对象给（`node_polity` / `apply_polity`）。
21. **切默认**：`[engine] backend = "cpp"`；tests/conftest.py 让扩展没编的机器上不依赖 C++ 的测试走 python。

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
P6b 的指数 ziggurat 表同样探出来（`probe_ziggurat.py` 一起写进 `ziggurat_tables.inc`）；泊松、不放回抽签照 distributions.c / _generator.pyx 同式，测试里逐位比。
本机（numpy 2.5.2、X86_V3 分派）实测 numpy 的 float64 exp / log / log10 / log1p / cos / sin / arctan / arctan2 / hypot / power 与 UCRT 的同名函数逐位相同（没有 AVX-512 的 SIMD 版），
BLAS 是 OpenBLAS 的 SkylakeX 内核（`blas_ddot` 照它的乘加次序）；换机器要重新验（DESIGN-NOTES 四点二十四）。

## 五、绑定接口（`skyisle_gen._core`）

两层：**步**（前端正式调用）与**公共件**（给 pytest 做同输入对照、给调试用）。数组一律 numpy、C 连续、行主序；传入时复制一份，返回的数组由 C++ 分配、numpy 持有。

| 函数 | 输入 | 输出 |
|---|---|---|
| `build_terrain(inp, planet, cfg, res_m=None, threads=0)` | inp：本群标量（节点号、seed、lat / lon、陆地、主岛、台面、岛龄、叠层、keel 余隙）；planet：板块网格（lat0 / dlat / lon0 / dlon + K + btype + lats / lons）、全体群的 lat / lon / 陆地、行星半径；cfg：展平的 [island] | dict：H / W / res_km / x0 / y0、height（f64，虚空 NaN）/ island_id（i16）/ cliff（bool）、各岛记录（面积格数、目标、岛心、台面、起伏目标、岸缘、峰、岛底、岛龄、类别、外框）、links、导水槽树、势力范围记录、masks_pos、轴向 / 边界核 / 类型、用时 |
| `generate(inp, planet, cfg, year=0, steps=5, res_m=0, threads=1)`（P6b） | inp 另加 ④ 的气候标量（temp / storm / window / season_range…）、pop（⑨，没有给 None）、people_per_arable_km2、capital（本群是邦都时 {state, state_pop, reformer}）；planet 另加 climate（④ 的 climate_grid 网格、band_local 的 eq_n / eq_s、倾角、热惯性常数、历法） | dict：terrain / hydro（同上两行，landcover 是最终值）、resources（terrain_zone、patch_id、resource、res_field [6,H,W]、occ_lab、deposits / occurrences / workings 记录（ASCII 代码）、occ_cells、geology 数）、climate（记录）、daily、weather（逐日数组 + 各季参数）、settle（settlements 的形）、settle_raster / settle_fields / settle_pop、各段用时 |
| `block_reduce(island_id, height, landcover, river, lake, f)`（P6b） | 群栅格 | 粗版一层：land / height / peak / island / landcover / water（dtype 同 lod._block_reduce） |
| `climate_only(inp, planet, cfg)`、`weather_years(inp, planet, cfg, rim_m, years)`（P6b） | 同 generate | 四季记录；多年逐日的季降水和 P[年, 季]、季雨日比例 F[年, 季]、f_rain_days |
| `weather_year(inp, planet, cfg, rim_m, year=0)`（粗版带天气，DESIGN-NOTES 四点二十七） | 同 generate；rim_m = 水系之后主岛的岸缘（island.json 的值） | climate（记录）、daily、weather（与 generate 的同形：一年逐日数组 + 各季参数） |
| `build_hydro(state, planet, cfg, threads=0)` | state：build_terrain 的数组与各岛的 rim / keel / age（**用 island.json 里已四舍五入的值**，与 Python 版同口径）、origin_km（同样是四舍五入后的）；planet：局地风网格、年长秒数；inp 的降水 / 海面温 / 直减率 / 有河 / 河级 / 可耕率 | dict：height（改后）、filled、flowacc、river / stream / lake、宽 / 深 / 漫滩 / 下切、recv_i / recv_j / route_h、landcover、arable、slope、各岛水系摘要、主岛盆地、河口表、河道折线、台面校正量 dz |
| `planet_stage1(cfg)` → `planet_stage2(cfg, P)` → `planet_stage3(cfg, seed, P, W)` → `planet_stage4(cfg, seed, P, W, I)`；`planet_run(cfg, seed)`（P6c） | cfg：`make_config(engine.planet_config(cfg))`（shared / skeleton / s01–s04 展平，有字符串键）；上游是前一步返回的不透明对象（或从产物读回的） | 不透明对象 `PlanetParams` / `Winds` / `Islands` / `Climate`；`planet_run` 返回四个的元组（游戏新建世界的路径，不经 npz） |
| `planet_json(P)`、`winds_arrays(W)`、`islands_arrays(I)`、`climate_arrays(C)`（P6c） | 各步对象 | planet.json 的 dict；各 npz 的数组（双精度原值，前端照 Python 版转 float32 再写）与摘要要的量（in_stack、n_exp / n_fallback / n_chord、f0、dt_s / n_steps） |
| `planet_from_json`、`winds_from(wind, bands)`、`islands_from(islands, plates, cand_edges, density_grid)`、`climate_from(wind_local, band_local, climate_grid, climate_islands)`（P6c） | 产物（npz 的数组 dict / json 的 dict） | 各步对象（上游命中磁盘缓存、换了进程时用） |
| `planet_view(P, I, C, cfg)`、`node_inputs(I, C, node, seed, cfg)`（P6c） | 行星层对象 | 第三层的 PlanetView（`Planet` 对象）与本群 NodeInputs（dict，同 `engine.inputs`；⑨ 的 pop / capital 由 `node_polity` 给） |
| `planet_stage5(cfg, P, W, I, C)` → `planet_stage6(cfg, seed, W, I, C, B, threads=1)` → `planet_stage7(cfg, P, I, C, B, R)` → `planet_stage8(cfg, seed, I, B, R, Ce)` / `planet_stage9(cfg, I, C, B, R, Ce)`（P6d） | cfg 同上（另有 s05–s09、slots、traits_manual）；上游是前面各步的对象 | 不透明对象 `Barriers` / `Routes` / `Centers` / `Diffusion` / `Polity`（⑨ 不读 ⑧） |
| `planet_run(cfg, seed, upto=4, threads=1, skip_diffusion=False)` | 同上 | upto = 4：(P, W, I, C)（P6c 的形）；5–9：再接 (B, R, Ce, D, Pol)；skip_diffusion：⑧ 不算、⑨ 照算 |
| `barriers_arrays(B)`、`routes_arrays(R)`、`centers_arrays(Ce)`、`diffusion_arrays(D)`、`polity_arrays(Pol)`（P6d） | 各步对象 | npz 的数组（双精度原值）与 json 要的原始记录（特征表带槽位序号、邦的记录、纪年、战线、开局候选；中文与 round 由前端） |
| `barriers_from(perm)`、`routes_from(routes, hubs)`、`centers_from(centers, prehist, regions)`、`polity_from(polity, polities)`（P6d） | 产物 | 各步对象（⑧ 没有下游读它，不读回） |
| `node_polity(Pol, node, cfg)`（P6d） | ⑨ 对象 | {pop（float32 的值）, people_per_arable_km2, capital（{state, state_pop（float32 成对求和）, reformer} 或 None）} |
| 公共件 | （P6d）`graph_dijkstra`、`graph_betweenness`、`graph_weak_components`、`rng_choice_noreplace_p`；`rng_raw / rng_draw / rng_choice_p / rng_choice_noreplace`（各分布，P6b 加 exponential / gamma2 / lognormal / poisson）、`np_quantile`、`np_interp`、`np_convolve_valid`、`blas_ddot`、`np_sum_f32`、`label_by_island`、`window_extrema`、`kmeans_split`、`crc32`、`np_sum`、`pyround`、`math_fns`（numpy 同式的幂与斜边）、（P6c）`rfft_lowpass`、`insolation_first_harmonic`、`planet_fractal_noise`、`planet_knn`、`median_f32`、`grid_interp`、`fractal_noise`、`label_components`、`largest_component`、`binary_erode / dilate`、`distance_bands`、`nearest_propagate`、`block_mean / any`、`upsample_bilinear`、`smooth121`、`laplacian`、`slope_deg`、`priority_fill`、`fill_iter`、`d8`、`d8_random`、`accumulate`、`island_shape`、`radial_profile`、`sculpt_island`、`territory_limits`、`nearest_fit`、`boundary_axis` | 同 Python 版的返回 |

配置用**扁平键值表**：前端把 `[island]` 段展平成 `{"layout.n0": 30.0, "territory.stretch": [1.0, 1.6, 2.4], "territory.enabled": 1.0, …}`（布尔 → 0 / 1，字符串跳过；
行星层（P6d）另把表的数组展成 `"slots.slot.n"` 与 `"slots.slot.<i>.id"` 这样的键：槽位表、政治性障碍、手工特征表），
C++ 按键取、缺键抛异常。以后游戏从 TOML / 行星包读同一张表。行星层（P6c）展平 shared / skeleton / s01–s04 全路径（`"s03.islands.n_islands"`），另有 `"str"`（`s01.calendar.mode`、`skeleton.g_anchor_edge`）。`[engine] threads`（默认 4）控制群内各岛并行的线程数，结果与线程数无关（每岛自己的随机流、写各自的格）。

**J（island.json）的拼装留在前端**：C++ 给原始的双精度数，前端照 Python 版的键序与 `round()` 位数拼（`engine.py`），所以两个后端的 island.json 同形。
P6b 的资源 / 聚落 / 四季记录量大，改由 C++ 直接给 JSON 形（`Json`，数已按 Python 的 round 舍好，因为 Python 版后面的计算读的就是舍过的值），字符串是 ASCII 代码，
前端 `decode.py` 译回中文；赋存区的长度 / 走向（`resources._shape`：np.cov 走 OpenBLAS 的 syrk、np.linalg.eigh 走 LAPACK，对称的小块协方差 ±1e−18 的符号决定走向 0° 还是 180°）
由前端按 C++ 给的格子（occ_cells）用 numpy 重算；C++ 自己的闭式主轴留给游戏用。
C++ 内部凡是 Python 版读的是「已四舍五入的 J 值」的地方（hydro 用的 origin_km、rim_m、keel_m，导水槽 Prim 堆里的 gap_km），C++ 也按 Python 的 `round` 取（`pyround`：printf 正确舍入 + strtod，与 CPython 的 round 同为「二进制值的精确十进制舍入、逢半取偶」）。

## 六、前端开关

- `config/default.toml` 新增 `[engine]`：`backend`（P6d 起默认 `"cpp"`）、`threads = 4`。读法与 `[island]` 同：default.toml ← run 的 config.resolved.toml ← `--set engine.backend=cpp`；
  `skyisle island … --backend python` 是它的简写（`run / stage --backend …` 同理）。第三层不进缓存 key；**P6c / P6d 起有 C++ 实现的管线阶段（①–⑨）在 cpp 后端下 key 另混入 "+cpp"**，
  python 后端的 key 与以前一字不差（旧 run 照旧命中；没有 `[engine]` 段的旧快照也按 python 算）。两个后端现在逐位相同，仍分开缓存：逐位只在本机验过，切后端要真的重算一遍（也免得拿另一个后端的产物去验它）。
- 分派点（P6a）：`island.build_terrain` 与 `hydro.build_hydro` 的第一行；（P6b）`island.generate` 的第一处（整群一次调 `_core.generate`）、`weather.multi_year_stats`（IS-daily）、
  `lod.build_lod` 的块降采样、`climate.classify_all`（全量季型）；（P6c）`s01_planet` … `s04_climate` 的 run() 第一行（产物与摘要由共用的 `_write` 写）；
  （P6d）`s05_barriers` … `s09_polity` 的 run() 第一行（同上），第三层 `engine.inputs(full=True)` 的人口与邦都（`node_polity`）。
  cpp 后端下各步的 C++ 对象按（run 目录, 阶段, 阶段 key）缓存在进程内（`engine.part`），下一步与第三层直接吃；没有就从该步的 npz / json 读回。cpp 后端但扩展没编：报错并给出构建命令（不静默退回，免得以为跑的是 C++）。
- cpp 后端下写产物、出图、校验、对照、操作台照旧是 Python，读的是同形的 g。

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
- P6b 加：村数 ±15%、户数相同（人口只读 ⑨）、资源处数（点 / 片 / 赋存区 / 采场）±20%、雨日比例相差 ≤ 0.05、季型相同；整套产物是否逐字节相同（island.json 去掉 meta.seconds / engine）；
  `--timing` 改量整群 generate（不写产物），`--no-python` 跳过慢的 python。

另：pytest 加 `tests/test_core_engine.py`（扩展没编就跳过）——随机数逐位、公共件同输入对照、小世界上两个后端的 build_terrain / build_hydro 对照、cpp 后端 IS-det、线程数不影响结果。

P6c（行星层 ①–④）的对照不走 `island compare`：
- 三 seed 各用两个后端跑整条管线（cpp 放 out/cpp-seedN），①–④ 的 npz 逐数组逐位、json 逐值、各步摘要比；⑤–⑩ 顺带比（仍是 Python，输入相同就该相同）；
- 三 seed 的 `check` 两个后端都要 0 硬 0 软（C1–C5、P1–P8、IL-*、SK-*）；
- 在 cpp 后端的行星层上再跑 `island compare` 抽样，确认第三层不受影响；
- pytest `tests/test_core_p6c.py`：公共件逐位、小世界两个后端的 ①–④ 逐位、key 分开、开关变体、第三层从 C++ 行星层对象取输入。

P6d（行星层 ⑤–⑨）同 P6c 的办法：
- 三 seed 各用两个后端跑整条管线（cpp 放 out/cpp-seedN，python 重跑放 out/py-seedN），①–⑨ 的产物逐数组逐位、json 逐值、摘要比；⑩ 比 json / 九格表与图的像素；
- python 后端重跑与改前的 out/seedN 逐位相同、阶段 key 相同；
- 三 seed × 两个后端的 check 全过；
- 切默认后 seed42 的岛群：cpp 在 out/cpp-seed42 上生成、与 python 在 out/seed42 上生成的逐文件比（2051、2050 与十几个群）；
- pytest `tests/test_core_p6d.py`：图算法与抽签逐位、小世界两个后端的 ⑤–⑨ 逐位、key 分开、planet_run(upto=9)、读回的对象接着算、四组开关变体、第三层的人口与邦都。

## 九、以后各期从哪接

- **P6b**（第三层其余）：已做（DESIGN-NOTES 四点二十四）。第三层现在整条在 C++：`island::generate(NodeInputs, PlanetView, Config)` 一次出地形到聚落，
  `block_reduce` 出粗版；Python 前端只拼 island.json、写产物、出图、校验。
- **P6c**（行星层 ①–④）：已做（DESIGN-NOTES 四点二十五）。`planet::run(Config, seed)` 一次出 ①–④，`planet_view` / `node_inputs` 直接给第三层；
  管线里各步照旧写 npz / json（前端），cpp 后端的阶段 key 另混入 "+cpp"。三 seed 的产物两个后端逐位相同、check 全过。
- **P6d**（行星层 ⑤–⑨ 与切换）：已做（DESIGN-NOTES 四点二十六）。`run_society(cfg, seed, World)` 接着 `planet::run` 出 ⑤–⑨，`apply_polity` 给第三层补人口与邦都；
  三 seed 的产物两个后端逐位相同、check 全过，**默认后端已切 cpp**。Python 算法当时没删（与原计划不同，主会话定的）：冻结成参考后端，
  等 Linux（ME Pro）上编过、在那边与参考后端对照过再删。**09-30 验过、删了**（DESIGN-NOTES 四点四十一 / 四点四十二）：`[engine] backend` 键、`--backend`、`island compare` 一起去掉，
  decode.py、⑩、check、九格表、出图、操作台留着；改动前后的回归改用 `docs/probes/diff_runs.py` 与改前的产物逐项比。
- 游戏（P6e）：`core/` 作子模块编进 GDExtension（CMake 目标 `skyisle_core`，静态库；`SKYISLE_PYTHON=OFF SKYISLE_TESTS=OFF` 就只编它，
  依赖只有 C++17 标准库与线程（`Threads::Threads`）和随仓库带的头文件 pocketfft；头文件入口 `skyisle/planet/planet.hpp`（①–④）、`civ.hpp`（⑤–⑨）、
  `view.hpp`（→ 第三层）、`island/generate.hpp`（整群））。剧情模式 NodeInputs / PlanetView 由行星包填；
  生存模式新建世界：`World w = planet::run(cfg, seed)` → `Society s = run_society(cfg, seed, w, 9, /*skip_diffusion=*/true, threads)`
  → `PlanetView v = planet_view(w.planet, w.islands, w.climate, cfg)`；某个群：`NodeInputs x = node_inputs(w.islands, w.climate, node, seed, cfg)`、
  `apply_polity(s.polity, node, cfg, x)` → `island::generate(x, v, island_cfg, …)`。全程不落盘、不经 Python。
  配置：C++ 的 `Config` 是扁平键值表，现在由 Python 前端展平 default.toml 给（`engine.planet_config` / `island/engine.flat_config`）；游戏需要一份不经 Python 的来源
  （例如生成器导出展平后的 JSON 随行星包带着，C++ 侧读进 `Config`），P6e 时定。记录是 JSON 形（`Json`），游戏按代码自己映射显示名。
