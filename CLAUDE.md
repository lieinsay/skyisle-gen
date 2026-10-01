# 空岛行星生成器 —— 工作约定（Claude 上下文）

这是 `docs/spec/12-扩散模型.md` 的实现：一颗浮空岛行星的地形、气候与文明生成器。
包名 `skyisle_gen`，命令 `skyisle`。**先读本文件，再读 `docs/DESIGN-NOTES.md`（决策与踩坑全记录）。**
上游规格随仓库带了两份快照：`docs/spec/12-扩散模型.md`（规格书）、`docs/spec/11-世界总图.md`（骨架定稿）。
其余上游文档（`01-设计铁律` 硬约束、`02-世界与地理` §3–5、`08-地区设计规程` 九格表格式、`04-社会与变迁` §3 四模式）
留在原 Zhouzhu 设计仓里，本仓库不含副本；下文与 `docs/` 里凡写 `docs/0X-…` 的，都指那边的文档。

## 环境与命令

- Python 3.12 的专用 venv（2026-09-30 起，DESIGN-NOTES 四点四十一）：`%LOCALAPPDATA%\venvs\skyisle\Scripts\python.exe`（PowerShell 里用 `$py = "$env:LOCALAPPDATA\venvs\skyisle\Scripts\python.exe"`；Git Bash 里 `"$LOCALAPPDATA/venvs/skyisle/Scripts/python.exe"`）。公用的 Python312 装着别的东西、numpy 是 1.26.4，**不要用它、也不要动它**。依赖仅 numpy + matplotlib（+ pytest；编 C++ 核心要 nanobind + CMake + VS 2022）；**numpy 钉在 2.5.2**（C++ 的逐位对照照它追的，别升别降）。**不引入 scipy/networkx/pandas。**
- **算法只在 C++ 核心库里**（`core/`；Python 参考后端 2026-09-30 删了，git tag `python-reference-final`，DESIGN-NOTES 四点四十二）：新机器先 `$py core/build.py`，扩展没编时报错。
- **ME Pro（Debian，无显示器，`ssh liein@10.8.0.12`）**：09-30 按新名字重装（四点四十一）——代码 `~/dev/skyisle-gen`（从 GitHub clone）、venv `~/.venvs/skyisle`（Python 3.13、numpy 2.5.2）、
  软链 `~/.local/bin/skyisle`、`~/.skyisle-env.sh`（`SKYISLE-DEV-ENV` 段，`~/.bashrc` 与 `~/.profile` 都 source；带代理与 `MPLBACKEND=Agg`）。C++ 核心已编（g++ 14、cmake 3.31、Ninja），
  删 Python 算法之前在那边验过：三 seed ①–⑨ 与 `island compare` 30 群两个后端逐位相同；和 Windows 只差浮点末位（两边的 libm 不同），离散结果全同。
  中文图标：`fonts-noto-cjk`（Noto Sans CJK SC 在字体回退表里）；装了新字体要删 `~/.cache/matplotlib/fontlist-*.json` 让它重建。`pipeline` 与 `serve` 不要同时跑；操作台绝不绑 `0.0.0.0`。
- 一律在仓库根下执行：
  ```
  $py -m skyisle_gen.cli run --seed 42            # 十步全跑（①–⑨ 约 3 s（C++），大头是 ⑩ 出图约 1 分钟；只改 [s0k] 的参数就从第 k 步起重算）
  $py -m skyisle_gen.cli stage 6 --seed 42        # 从第 6 步强制重算
  $py -m skyisle_gen.cli check --run out/seed42   # 八条验收（P1–P8）+ 气候 C1–C5（C5 四季分明）+ 铁律自检 + 骨架/历法校准（exit 0/1/2）
  $py -m skyisle_gen.cli viz all|wind|islands|scale|routes|polity|isogloss|slot <id>|trait <id>|distance <node> --run out/seed42
  $py -m skyisle_gen.cli polity --run out/seed42  # ⑨ 政治层摘要：宗主 / 变法之国 / 兼并纪年 / 最大诸邦 / 开局候选
  $py -m skyisle_gen.cli probe node <id> | edge a b | path a b --mode m | trait <id> --node j
  $py -m skyisle_gen.cli ninegrid --run out/seed42 [--region K]
  $py -m skyisle_gen.cli island 1165 --run out/seed42 [--year 0] [--res 100] [--export DIR] [--set island.x.y=v]
                                                  # 第三层岛群生成器：out/seed42/islands/1165/（约 5–15 s；不进管线、不回灌）
  $py -m skyisle_gen.cli island check 1165 --run out/seed42   # IS-* / SET-* / RES-* 校验（含重跑比哈希）
  $py -m skyisle_gen.cli island batch --run out/seed42 --sample 30   # 分层抽样批跑 + 校验 → islands/batch.json
  $py -m skyisle_gen.cli island floats --run out/seed42 [--jobs 28]  # 浮高的全行星统计（只跑布局 + 地形，cpp 约 5 分钟）→ islands/float_stats.json / .npz；不在标定区间退出码 1
  $py -m skyisle_gen.cli island stats --run out/seed42   # 全量季型统计（只算气候，8000 群 4 s）→ islands/season_stats.json；操作台气候视角「季型」着色读它
  $py -m skyisle_gen.cli island lod --run out/seed42 --lod-res 2000,1000 [--nodes a,b | --near 2051 --radius 600] [--jobs N] [--force] [--no-weather] [--year 0]
                                                  # 岛群粗版（远处看的低模）：原生分辨率生成再按块降采样 → islands_lod/<分辨率>/<节点>.npz + index.json；
                                                  # 默认顺带出第 --year 年的逐日天气：同一 npz 的 weather_<列>（与 weather_y0.csv 逐行相同，外加 temp_rim_c）+ weather_meta（气候参数头，四点二十七）；
                                                  # 已有的跳过（不带天气 / 年份不对的重跑）；全行星 2 km + 1 km 在 cpp 下 22 进程 8.5 分钟（Zhouzhu 的 tools/export_planet.py 读它打行星包）
  $py core/build.py [--test] [--debug] [--clean]  # 编 C++ 核心库 → skyisle_gen/_core.*.pyd（要 pip install nanobind；Windows 自动进 VS 2022 x64 环境，CMake + Ninja，约 1 分钟）
  $py docs/probes/diff_runs.py out/seed42 out/verify42          # 改动前后的回归：①–⑨ 的 npz 逐数组逐位、json 逐值、md 逐字（--tol 1e-9 放过跨平台的末位）
  $py docs/probes/diff_runs.py 旧/islands/2051 out/seed42/islands/2051   # 岛群产物：npz 逐数组、json 逐值（去掉用时）、png / csv 逐字节
  $py -m skyisle_gen.cli town site 2051 --run out/seed42 --site 村037 --style 华北集村   # 聚落营建器（独立工具，PLAN-TOWN）：岛群里一个聚落的地细化到 1–2 m 并营建
                                                  # → islands/2051/town/村037-华北集村/（site.npz / plan.json / plan.png / plan-detail.png / plan.svg / style.resolved.toml）；不给 --style 只出地面
  $py -m skyisle_gen.cli town synth --terrain 河谷 --style 江南水乡 [--operator waterfront] [--scale 村|小庄|宅院] [--households 60] [--seed 1] [--lat 35]
  $py -m skyisle_gen.cli town gallery --terrain 河谷[,平原] --styles all [--operators each] [--households 40]   # 画廊 → out/town/gallery/…/gallery.png + gallery.json
                                                  # 十种合成地形 → out/town/synth/；--heightmap h.png --res 1 用外部高程图；town style list / show 华北集村
  $py -m skyisle_gen.cli serve                    # 3D 操作台 http://127.0.0.1:8642/（完全离线）；岛群调试台 /island.html?run=seed42&node=1165；营建调试台 /town.html?run=seed42&node=2051&site=村037
  skyisle serve --host 192.168.0.116,10.8.0.12 --no-open   # ME Pro 上这样起（--host 可多地址；拒绝 0.0.0.0）
  $py -m skyisle_gen.cli viz web --run out/seed42 # 单文件 viewer.html（内嵌 globe.gl）
  $py -m pytest tests -q                          # 246 个测试，约 5 分钟（test_island / test_core_engine / test_core_p6b / p6c / test_core_float（三 seed）跑 1600 岛的小世界到 ④，p6d 与 test_pipeline 到 ⑨；
                                                  # 扩展没编时只剩 test_core 里不靠 C++ 的 23 个（静态断言、配置校验），其余整个文件跳过；四点四十二起逐位对照改成只测 C++ 的性质）
  $py docs/probes/transect.py out/seed42/islands 6615,6610,6329,6073,6072,5766,5498,2051 --run out/seed42
  $py docs/probes/works_view.py out/seed42/islands/6329 --auto big|head|patch|ruin [--out 图.png]   # 水利俯视图（P6）：灌田最多的大堰 / 渠首的渠网 / 最大的一片圩田 / 废村旁的废渠塘（也可 --big N / --head N / --patch N / --ruin N / --rows --cols）；
                                                  # P6b 起画管它的村（渠首、圩的虚线）、圩村（绿边）、废渠 / 废塘（灰）；四点四十起画邑级大堰（橙）与分水口（青）
  $py docs/probes/market_view.py out/seed42/islands/6329 [--seat [--pad 150]] [--out 图.png]   # 镇与航船俯视图（P7）：大泊场、镇与朝泊场的街、邑治、航船线、中转站（整群或邑治一带）
  $py docs/probes/sample3.py                 # 给 daywater / koppen 之类量具取样：三 seed 各按降水取 4 个节点生成岛群产物（四点五十）
  $py docs/probes/daywater.py [run ...]      # 量「季相与水情」日层的溪涧干季收缩（把 island.html 的日层模型在 Python 里复现）：逐节点的一年最干那天还剩多少溪涧有水、A_REF 取 1.0/1.4/2.0/3.0 各是多少（四点五十）
                                                  # transect：量剖面（Zhouzhu PLAN-LAND 第二节同口径：地形 / 河 / 湿地、宜垦 / 已垦 / 撂荒、小岛有没有人、有人用没人住的岛、水利、镇 / 邑治 / 航船 / 中转站、
                                                  # P6b 的水利离村多远（离最近的村 / 离管它的村）与原始地貌 / 人工改造），直接读岛群产物
  ```
- **C++ 核心库（`core/`，行星计划 P6；设计稿 `docs/PLAN-CORE.md`，DESIGN-NOTES 四点二十三 – 四点二十六、四点四十二）**：第三层全部（P6a / P6b：`island.generate` 一次调 `_core.generate` 算完）
  与行星层 ①–⑨（P6c / P6d，`core/planet/`）的算法都在 C++ 里；s01–s09 的 run() 调 C++，npz / json / md 与摘要由 Python 的 `_write` 写（中文名、round 的位数都在那里）；
  ①–⑨ 的缓存 key 固定混入 "+cpp"（沿用 cpp 后端时的 key，删 Python 之前的 run 照旧命中）；第三层的 PlanetView / NodeInputs（含 ⑨ 的人口与邦都）由 C++ 从各步的产物对象直接给
  （同进程跑过就不读 npz）。Python 只剩前端：命令、配置、写产物、decode.py（C++ 的 ASCII 代码 → 中文）、check、九格表、出图、操作台、⑩。
  **Python 参考后端 2026-09-30 删了**（ME Pro 上验过逐位相同之后；git tag `python-reference-final` 是删之前的最后一版，C++ 注释里「xxx.py 同式」指的是那一版）。
  改了 `core/` 要重编（`$py core/build.py`，增量几秒）；**有进程还在用 `_core.pyd` 时重编会链接失败**（Windows 锁文件）。Rider / 别的构建目录不要和它同时编同一个 `build/core`。
- 验收基线：**seed 42 / 7 / 2026 三个种子 `check` 必须全过（0 硬项 0 软项）**，改动核心公式或默认参数后都要重跑这三个。
- PowerShell 向 `python -c` 传含引号的代码会被破坏：写成脚本文件再跑。
- 产物目录 `out/` 已 gitignore；`config.resolved.toml` 是 `check/viz/probe` 读取配置的来源——改了 `[check]` 阈值要先 `run` 一次刷新它。
  岛群生成器的 `[island]` 也是「default.toml ← 该 run 的快照 ← --set」：**改已有键的默认值，旧 run 仍用快照里的旧值**（先 `run` 刷新，全命中缓存、不到 1 s）；
  **键的含义变了就改键名**（旧键留在快照里无害），否则旧 run 会拿旧含义的数去用。快照写出时中文键要加引号（`dump_toml` 曾把 `汇聚 = 4.0` 写成裸键，读不回来）。
- 提交信息用中文；`out/`、`out-*/` 不提交。

## 架构速查

```
skyisle_gen/
  pipeline.py    阶段注册、缓存 key 链（config[s0k]+shared+skeleton+seed+STAGE_VERSION[+"+cpp"]）、产物 IO
  engine.py      （P6c / P6d）C++ 桥：行星层配置展平（shared / skeleton / s01–s09 / 槽位表 / 手工特征表；表的数组展成 key.n 与 key.<i>.<子键>）、
                 阶段 key 的 "+cpp"；各步的 C++ 对象按（run, 阶段, key）进程内缓存，没有就从 npz / json 读回（part；⑧ 不读回）
  config.py      TOML 加载/深合并/--set/校验（通过率∈[0,1]、r≤0.98、半衰序 daily≤trade≤migrate≤envoy、eps0>0；engine.backend 只认 cpp）
  stages/s01…s10 十步；①–⑨ 的 run(ctx) 调 C++（core/src/planet/stage*.cpp）、`_write` 写 npz/json/md + 摘要，⑩ 全在 Python
                 ⑨ s09_polity 政治层（第四批 R7：人口、诸邦、采邑、名分/附庸、变法与兼并史），⑩ s10_output 输出（原 ⑨）
  localwind.py   只剩八条局部带界的键与取值（EDGE_KEYS / edge_lats，⑤ 与 check 用）；板块（第三批 2）、岛对风的扰动（第三批 3）、水汽降水（第三批 4）都在 C++（stage3 / stage4.cpp）
  graph.py       CSR、Dijkstra(heapq)、Brandes 抽样介数、弱连通分量 —— check（P6 的反事实介数）/ probe / 操作台的路径用的纯 Python 版，注意 inf 比较；管线里用的是 C++ 版 core/include/skyisle/planet/graph.hpp
  weights.py     w_m = λ_ref·cost_m + L_m（L = −ln perm，perm=0 → inf）
  culture.py     World 惰性读取；槽位份额（含本地行）；TV 文化距离；同言线边集
  skeleton.py    骨架第二版的几个读法（D 纬度域、文明核心窗、带界缩放、年长）给 check / 九格表 / 第三层；G 锚定、密度剖面、季节强度、历法 ↔ 轨道（R1）的算法在 C++（core/src/planet/）
  polity.py      政治层产物的只读封装（Polity：邦名/状态/探针行；print_summary），ninegrid/probe/web 共用
  geology.py     地质表现层（R4）：③ 板块网格按节点采样 → 九格表 ① / 探针 / 操作台的叙事文本，不进推导（原则甲）
  check.py       P1–P8 + C1–C5 + IL-*（铁律）+ SK-*（骨架校准，warn-only）
  ninegrid.py    九格表草稿（RegionData 聚合 + build_region_md + lint）
  viz.py / probe.py / web/(server.py bundle.py static/index.html static/vendor/globe.gl.min.js)
                 操作台数据通道：/api/world、/api/fields、/api/grid（② 风 / ④ 气候的 1° 网格场，R2）、/api/texture；单文件版全部内嵌于 INLINE
                 /api/island?run=&node= 按需生成岛群并返回摘要，/api/island/preview 取总览图（探针折叠区「岛群生成器」；单文件版不支持）
                 **岛群调试台** `static/island.html`（`/island.html?run=&node=[&year=]`，探针里有链接）：2D canvas 图层（地形 / 晕渲 / 地表 / 坡度 / 汇流 / 岛号 / 当日海拔温度 / 地形区 / 岩性（B2）/ 水位埋深（B，四点四十九）/ 资源分布（主导，或某一类的赋存品位）；特殊的山（B3）、泉线与瀑布（C3 / C5）叠加层；`&base=zone|resource&rk=stone&res=1&fly=行,列,缩放&day=N` 可直接打开；拉远时季相层 / 河道矢量自动降级）、
                 滚轮缩放拖动、悬停读格（高程 / 坡 / 汇流 / 地表 / 水与河宽水深、谷底宽与限制度 / 地形区 / **水位与埋深** / 凝结水与云雾林 / 主导资源与各类赋存品位 / 当日温度）、资源点位（采场 / 点 / 片，拉近标赋存区）与漫滩叠加层、主岛河流表（点按钮飞到河口）、约束对照、四季表与图、逐日天气图 + 日期滑杆 / 播放、改年份重生成、`island.*` 参数覆盖重生成；
                 「季相与水情」日图层（积雪 / 雪线、植被枯荣、作物阶段、溪涧按基流水库逐段断流 / 接回、河道涨水漫滩、冰按度日封冻 / 开河、云海漫顶；河道永远画，断流是干河床，四点二十）与「天气特效」（雨雪风暴云雾风粒子）都在浏览器里按逐日天气推，不改产物（DESIGN-NOTES 四点十四）；
                 数据通道 /api/island/data（island.json + climate.json 含 weather.days）、/api/island/raster（terrain.npz 定型数组 base64，> 160 万格抽稀）、POST /api/island/regen；
                 左栏「营建聚落」与双击村子 → /town.html
                 **营建调试台** `static/town.html` + `town_api.py`（PLAN-TOWN 第五步，DESIGN-NOTES 四点三十二）：/api/town/styles、/api/town/style?name=、/api/town/sites?run=&node=、
                 POST /api/town/plan {source synth|site, …, style, operator, sets, save} → 底图 PNG + 抽稀栅格 + 方案（plan.json 同段）；地面按来源与窗口系数缓存 8 份，营建加锁
  island/        **第三层岛群生成器**（PLAN-ISLAND，DESIGN-NOTES 四点十四）：`skyisle island <节点>`，按需生成、不进十步管线、不回灌
                 （stages/ 与 check/ninegrid/polity/culture 不得 import 它，pytest 与 IS-iso 有静态断言）
                 **产物有版本戳**（四点四十八）：island.json 的 `meta.stamp` = `ISLAND_VERSION` + `[island]` / `[engine]` 配置 + 上游 ①②③④ 的 stage key + seed；
                 控制台与 `products_stale` 拿它对盘上的产物判新旧，对不上就重生成（`--set` 的一次性覆盖不进戳）
                 **下面各条的算法都在 C++（core/src/island/ 的同名 .cpp）**；Python 的同名模块只剩前端（类表与中文、备注、island.json 摘要、写产物、出图），
                 layout / territory / river / tiers / compare 的 .py 已删（四点四十二）
                 __init__  island_config（[island] 段：默认值 ← run 的 resolved ← --set，不进缓存 key）、ISLAND_VERSION / island_stamp / products_stale（产物新旧）、_node_inputs、build_terrain、generate
                 grid      局部分形噪声（特征尺度以 km 给）、行程并查集连通分量、形态学、块均值 / 双线性（C++ 的 grid.hpp）；Python 只剩平移、膨胀（check 用）与 PNG 写出
                 layout    5.1 岛数（n0=30 × 陆地^0.35）、Zipf 大小（总和严格 = area_km2，主岛最大）、角向半径剖面放置（主岛引力、板块走向拉长）、各岛台面高度与目标起伏（岛龄 × 面积^0.3，另一条随机流）、
                           浮高 float_offsets（四点二十八：其余岛按岛龄整座上下平移 δ，又一条随机流；往下的等岸缘拟合出来再按离 rim_floor_m 的余量缩，平移在 build_terrain 里、水系之前）、
                           短渡（P5 起没有索桥与导水槽，四点三十六）
                 terrain   5.2 岛形（椭圆 + 域扭曲 + 面积二分反解）、岛龄基形（**拱** / 脊 / 台地，四点三十五：新岛是从海底挣脱出来的拱，不是火山锥）+ 幂次定测高曲线、
                           **多核嵌合**（multicore_spec / _multicore_form：汇聚带一部分大岛两三个核，随机流 island:<节点>:cores:<岛号>；拟合后量各核载荷 → island.json 的 cores，cores_json 两后端共用）、
                           **隐式河流功率下切**（B1 起在原生分辨率上、≤ `erosion_max_cells` 2048 格，每轮全岛填洼；只切汇流 ≥ 0.3 km² 的河道格、坡面靠岩性的坍塌角；随机流向 + 平滑去方格纹）、
                           仿射拟合（陆地中位 = 台面、峰 − 岸缘 = 目标起伏；层面跟着同一个仿射变）；priority_fill / d8 / d8_random / accumulate（5.3 共用）
                 strat     （B2，四点四十六）岩层：每岛一套层面（构造顶面 strat_top、骨架顶面 skel_top）与层序（沉积盖层 = 硬石灰岩 / 软泥灰岩互层、老岛的硬盖、辉长岩、蛇纹岩、浮石骨架），
                           lith_at 按「在构造顶面下多深」给出露岩性；侵蚀每轮按岩性取可蚀性、坍塌角、坡面扩散（硬层成崖、软层成坡）；冠顶剥去的深度按岛龄插值（m）；
                           随机流 island:<节点>:strat:<岛号>；水系之后出 terrain.npz 的 lith（lith.png），地表（蛇纹岩秃山）、资源（石料岩性、盐、溶洞、铬镍、浮石露头）都读它
                 landforms （B3）特殊的山，按成因条件出、不设配额：褶皱（平行岭谷）与掀斜断块在 sculpt_island 里；峰林、冰川 / 冰斗 / 角峰、临空断山在 landform_pass（浮高之后，按本岛的气温）；
                           天上的山在 build 里只加标签；资源之后识别锯齿峰、蛇纹岩秃山、方山 / 孤丘、穿山天窗与崖层 rockwall_m / rockwall_dir → island.json 的 landforms；
                           Python 的 landforms.py 只有中文名与说明
                 coast     （B4）带符号的亚格岸距 coast_dist_m（岛形连续场 f = τ 的 marching squares 线段 + 到线段的精确距离；陆正、虚空负，m）
                 hydro     5.3 河（主岛按 has_river 调阈值）/ 溪涧 / 湖 / 河口盆地、地表 12 类、「上等地」按适宜度分位取到 arable_frac（g["arable"]，资源层避开它；
                           P5 起不是已垦）、宜垦 cultivable（farmland.cultivable_land 的钩子）；
                           流向在「路由面」上算（填平面 + 弯曲噪声 + 朝岸缘微倾：河在缓坡上蜿蜒、不贴崖边平行跑），湖与抬洼仍按原填平面；
                           四点三十五：**局地雨** local_rain（海拔 × 山脉尺度迎风坡、按全群均值归一 → rain_mm；流量 = 按径流深加权的汇流，权重取整到 1/16 mm；**A5 起径流深 = 局地雨 × 按格的 Budyko 径流系数**（傅抱璞式 w = `budyko_w` 2.6，Thornthwaite 潜在蒸散按本格高度的季温与季中昼长；
                           旧的全群 `runoff_coef` 0.45 作废，大堰的水按上游的径流累计 runoff_acc，terrain.npz 多 runoff_mm）、
                           **谷收拢** capture_rivers（走水之前：大谷把 reach 以内小沟的上半截抢过来，切直沟只压低地面；多核岛与新岛不收 / 少收）、
                           **湿地**（方窗最大汇流 × 雨³ × 离崖缘 × 洼 / tan(周围最陡的坡)）
                 river     5.3b 河道成形（DESIGN-NOTES 四点十六、**四点四十七**）：水力几何 w = 5·Q^0.5、d = 0.35·Q^0.4，真实比例（C1 删了 `width_scale` / `depth_scale`，旧 ×8 / ×3）→ 河宽 ≥ 2 格时加宽；
                           河床下切并向下游单调、至少按 `min_grade` 1 m/km 往河口降、河口切豁口成瀑布；**河道格的 height 是平岸水面**（C1，河床另存 bed_m；只有宽够一格的 river_water 记成水面，
                           更窄的河道格地表是岸上的、可以种）；两岸压成「谷底 + 谷坡」剖面（C2）：谷底宽 = 河宽 + 50·A^0.4 × 岩性 × 岛龄 × 比降系数 1/(1+(S/2%)³)，
                           限制度按 谷底宽 / 河宽（≤ 3 峡谷、≥ 15 开阔），谷坡在该处岩性的坍塌角与 15° 之间；谷底 = 平岸水面 = 漫滩；溪涧浅切、按比降接到干流上
                 groundwater（C4 / C5，四点四十七）**集水核**：强度 = cbrt(高出岸缘的山体 km³) / 10，凝结水 = 局地雨 × min(1, 强度 × 高出岸缘的比例^1.5 × 迎风 × 湿度)，**只进水账**
                           （runoff 含、rain 不含）；云雾林；热的千分之一进岩体 → 大核山（强度 ≥ 0.25）的核山温泉
                 water     （B+A，四点四十九）**水位面**（潜水面）：∇·(T∇h) = −R 的离散形式在 1/gw_coarse 的粗格上 SOR（排水口 = 河道 / 溪涧 / 湖 / 岸缘，
                           固定水头 = 地表；水位高过地表钉回地表 = 渗出面；不低于骨架顶面 + 最小含水厚），双线性插值回原分辨率、排水口在细格上钉回自己的地表
                           → terrain.npz 的 **wt_m**（水位，零点口径）与 **wt_depth_m**（埋深 = 地表 − 水位）：井打多深、挖到哪层见水、泉在哪都读它
                 groundwater（C5 的泉线，B+A）**流向按水位面的梯度**（不再按地表）；补给进了河道是河的基流，走到岸边的成**崖壁泉线**；
                           按 1 km 一段（springline_seg_km）分三等——**弥散渗出 / 泉 / 崖瀑**（出水 ≥ 本岛出口段分位 spring_pct / spring_fall_pct
                           **且**含水层厚度 ≥ spring_min_aquifer_m 才算泉 / 崖瀑，A）；泉的密度、种类（岩溶泉 / 接触泉 / 裂隙泉 / 渗水泉）、出水按岩性与补给
                 rivernet  （C3，四点四十七）河的数据：天气之后（有聚落就在之后）算——干支分段（从出口按汇水最大的一支往上追）、沿程每点（年均 / 平岸流量、平岸宽深、比降、
                           D50（平岸 Shields 数）、平面型（Kleinhans & van den Berg）、悬沙（BQART）、左右谷底宽、限制度）、瀑布与跌水、每条常年河一个流域的逐日径流指数
                           （积雪按高程分带、度日融雪、快流与基流两个线性水库）→ rivers.json 的 segments / falls / basins
                 resources 5.3c 地形区（高山 / 山地 / 丘陵 / 台地平原 / 河谷 / 崖缘 / 水域）+ 16 类资源分三形态（四点十七 / **四点二十一** / **四点三十四**）：
                           **点**（泉眼、温泉、洞穴、盐泉、贝壳化石）与**片**（林木、泥炭芦苇、浮石露头、鸟粪石）记在 deposits，片占的格写 patch_id（面积一律按它数：sync_resources）；
                           **散**（金属矿、石料、黏土、砂砾、砂金、硫磺、岩盐）= 赋存场 res_field（uint8 [7,H,W] 品位，可叠）→ 赋存区 occurrences（≥ occ_thr 的连通块）→ 采场 workings
                           （矿坑 / 硫磺坑 / 盐井在这里挑；采石场 / 土坑 / 采砂场 / 淘金点在聚落之后按村挑，tiers.village_workings）。
                           **没有火山**（P3，四点三十四）：矿是岛从海底带上来的（热泉硫化物 铜 / 铅锌 / 金银 / 铁 / 锡，蛇纹岩 铬 / 镍，锰结核），硫磺是海底热泉的、温泉是余热，都只在新岛；
                           熔岩管 → 骨架空洞（浮石露头旁）；岩层按剥蚀指数（高出岸缘到峰高的比例 + 岛龄偏移）分沉积盖层 / 辉长岩 / 蛇纹岩——石料岩性（海相石灰岩 / 辉长岩 / 蛇纹岩）、
                           岩盐盐泉与老岛溶洞只在沉积盖层、贝壳化石在海相石灰岩的石料区里；文字里不许有「火山」「熔岩」（RES-geo 查）。**岩类（矿 / 石 / 硫磺）只在岩类可放区**：
                           山地、高山、裸岩 / 高山草甸、丘陵且坡 ≥ 8°，林坡算（采场格改裸岩），平地林、耕地、湿地、漫滩不算；沉积类赋存可压田，坑不上田不上林（用户拍板）。
                           金属矿是顺板块走向的矿化带、板块内部几乎没有；砂金 = 砂砾 × 上游金银 / 铜矿化带的平均品位（hydro 存了群栅格的 D8 下游 recv_i/j 与路由面 route_h）；
                           浮石 = 岛体，崖面与峡谷壁按露头率露出、可开采不影响浮空。resource（uint8）只是显示用的主导类（稀的盖常的）；→ resources.json / png、terrain_zone.png、preview_resources.png
                 climate   5.4 四季：降水 = 年均 × 季数 × ④ 的份额 precip_share（A5）；风暴 / 窗口 / 风按 ④ 的每季带界位移 season_shift 取样再缩放到年均（旧 k_shift 作废）；温度 = 年均 + season_range/2·cos(相位 − 滞后)；季型分类命名
                 weather   5.5 逐日：马尔可夫晴雨 + 伽马雨量（风暴日计入预算）、风暴事件、AR(1) 风温、云海漫顶、岸缘 ≤ 0.5 °C 记为雪（小雪 / 大雪 / 暴风雪）；multi_year_stats 供 IS-daily
                 output    5.6 island.json / height.png(16 位) / landcover.png / water.png / arable.png / terrain.npz（含 w_ch_m / d_ch_m（河道 = 平岸宽深）/ floodplain / terrain_zone / resource / res_field / patch_id）/ climate.json / weather_y<年>.csv / preview.png（总览）/ preview_main.png（主岛放大）；
                           water.png 值 6 = 漫滩；河道格的 height 是河床，水面 = 河床 + 水深；rivers.json = 河道中心线折线（调试台 / preview 按河宽画平滑矢量，栅格上河只有一两格宽）；调试台拉近后停手会「细化渲染」当前视口（连续量双线性、分类图扰动 + 加权投票，纯显示）
                 settle    聚落（PLAN-SETTLE，DESIGN-NOTES 四点十五 / 四点十八 / 四点三十六 / 四点三十八）：人口只读 ⑨ → 户（10% 非农）；已垦（farmland：好地先占 + 定居门槛、撂荒、废村）
                           连通块切田块（k-means 按 80 户地量分；P6b：切好后圩田的格拿出来按 3 × 3 圩一组另成田块）；村址评分；圩田的田块挂到走得到的村上或自成圩村（polder_villages，P6b）；专业聚落与住法（常住 / 工棚 / 季节住）；P7 的大泊场、中转站、镇与航船、邑治（market）；
                           蓄水池 / 取水点；没人住的岛有人用（放牧 / 庙 / 墓岛；烽火台 P7 起归中转站）；
                           前哨、三个主家候选、都与城（城居人口 = 本邑 × 城居率 + 邦 × 集聚率，郭 = 同岛或船程 3 km 内，仓城 = 主泊场，祭台）；荒地归谁 → settlements.json / png
                 farmland  （P5，四点三十六）宜垦 cultivable_land（hydro 钩子）/ 已垦 fill_cultivated / 废村、住法、有人用、荒地归谁（C++ farmland.cpp 与 settle.cpp）；
                           P6 起 fill_cultivated 分两遍：第一遍之后挑人口压力到了的湿地排干（waterworks.polder_plan），第二遍圩田的格先占（额度之内）→ polder_id
                 waterworks（P6，四点三十七；**四点四十起分级**：`big_plan` 邑级大堰在 farmland 好地先占之前定灌区（适宜度 × (1 + big_suit_gain)），build_waterworks 最前面修大堰的渠与每个用水的村的分水口、邑管 → big_works；
                           `village_works = false`（默认）时下面的村级的渠首 / 渠 / 塘 / 废水利都不出，圩区照旧；村级的归营建器）水利：圩田（polder_plan：湿地周围 2 km 的平地种了过半才排干，600 m 一格一圩）、谷口的渠（build_waterworks：渠首从高往低挑在常年河 / 大溪涧上，
                           渠 = Dijkstra 最省工的路合成的树、划不划算按新接的渠长 ≤ 0.5 km + 3 km × 田的面积）、村塘 / 山塘 / 圩塘 / 堰塘、渠首闸 / 圩闸 / 排水闸、纵浦横塘与排水渠
                           → settlements.json 的 waterworks；C++ waterworks.cpp 逐位同式（Dijkstra 按 (工, 格号) 出堆、长度按直步 / 斜步计数）；
                           **P6b（四点三十九，L30 / L31）**：每处记管它的村（village，都在 manage_walk_km 2 km 内）——渠首一村一堰（只灌那个村的旱地田、Dijkstra 只走它走得到的格、灌区按格算、隔 1.5 km），
                           圩 / 圩塘 / 圩闸归种那组圩田的村，纵浦横塘按两旁的圩分段，排水渠截短；废村旁的废塘与（照活村的规矩找得到时）废渠首 / 废渠（abandoned）；
                           landuse_layers：terrain.npz 的 landcover_natural（没有人以前的地表）与 landuse（人工改造码 0–7）
                 market    （P7，四点三十八）泊场先于镇：大泊场 harbors（坡 ≤ 4° 的空平地开运算、1 km 方窗够 0.5 km²、隔 5 km，能停多少船）；中转站 relays（⑥ 的邻边按方位合成口子，
                           落在朝那边最外、有平地能存水的岛上：关卡 / 过夜 / 候风 / 避风 / 换船，群边高处的烽火台也在这里；户从非农户里出）；镇 towns（本岛走路 6 km + 跨岛只算航船够得着的，
                           船程 = 直线 × 风的系数，挨着大泊场加分，按还没被服务的户贪心）；航船 boat_lines（按方位扫一圈分线、从最远的村开到镇的泊场）；邑治（航船汇得最多、靠大泊场）；
                           node_routes 读 ⑥（C++ planet::apply_routes / _core.node_routes 同值）；C++ market.cpp 逐位同式
                 tiers     聚落层级（四点十八）：**飞船取代车船、随处可停 → 没有码头**；专业聚落（矿镇 / 浮石采石村 / 窑村 / 烧炭营 / 温泉地 / 盐井村，按资源量、封顶非农 40%）、
                           每个村 / 专业聚落旁一块泊场（船台，记能停几条船；镇在 market）、村周开垦（林地 → 草坡 / 灌丛薪炭林，林场 patch_id 划掉）、
                           村的采场（开垦之后：石 / 土 / 砂就近取，先合用 1.5 km 内已有的，再在 5 / 2 / 2 km 内按品位 × 距离挑；矿镇矿村改读金属矿赋存区）
                 territory 势力范围（四点二十二）：与每个邻群按等效半径 √(陆地/π) 分界、各退 gap/2；build_terrain 的 _fit_territory 照旧摆、越界了才重摆
                           （主岛挪进来 → 转向 → 拉长；其余岛带约束重摆），没越界的群产物逐字节不变
                 lod       岛群粗版（四点二十二）：原生分辨率跑布局 + 地形 + 水系再降采样——不能在粗分辨率上生成，布局随分辨率变；
                           `--weather`（默认开，四点二十七）接着算四季与一年逐日天气（不跑资源：天气只看行星层输入与主岛岸缘），存进同一个 npz 的 weather_<列> + weather_meta
                 floats    `island floats`（四点二十八）：全行星每群只跑布局 + 地形，收全部非主岛的浮高、Δ岛龄、岸缘，对标定区间（|δ| 中位 300–600 m、p90 0.8–1.2 km、
                           最高 ≥ 1.3 km、往上 60–80%、与 Δ岛龄秩相关 ≤ −0.4、岸缘 ≥ rim_floor_m）
                 check     第六节 IS-area/surface/arable/river/channel/water/season/float/link/terr/det/iso（硬；water 是 C6 的水账闭合）+ IS-valley（软，谷底宽合理）+ RES-site/occ/work/geo（硬）/ RES-quarry（软）+ IS-daily（软，600 年）+ SET-pop/field/site/land/town/home/farm/use/works/market/nature（硬，nature 是 P6b 的原始 / 人工对得上）/ SET-water（软）；batch 分层抽样批跑
                 engine    C++ 桥（四点二十三 / 四点二十四）：`generate`（整群一次调 `_core.generate`）、`build_terrain` / `hydro.build_hydro`（粗版、浮高统计）、
                           `weather.multi_year_stats`（IS-daily）、`lod` 的块降采样与天气（`_core.weather_year`）、`climate.classify_all`（全量季型）；
                           把原始的数拼回前端要的 g 与 island.json（键序、round 位数与删掉的 Python 版同形）
                 decode    （P6b）C++ 记录里的 ASCII 代码 → 中文、带数的备注用 f-string 拼；赋存区的长度 / 走向按 numpy 的 cov / eigh 重算
  town/        **聚落营建器**（PLAN-TOWN，DESIGN-NOTES 四点二十九 – 四点三十三）：**优先次序地形 > 合理 > 风格（用户定，plan_site 放宽地面上限）**；**只给用户做参考，暂不接进游戏（用户定；Zhouzhu 不读它，用户开口前别往游戏里接）**；独立工具，只读岛群产物、不回写；任何别的模块不得 import 它（test_town 静态断言）。
                 算法只有 C++（core/.../town/：geom 几何、raster 哈希噪声 / 采样 / 精确欧氏距离 edt、site 场地——岛群窗口细化 / 十种合成地形 / 外部高程图 → 统一的 Site；
                 style 风格的强类型结构、analysis 场地分析、orient 朝向规则链、network 路网与 A* 寻路、plan 编排 plan_site、contour 等值线，src 里另有十个形态算子 op_fishbone / op_organic / op_street（街村、林地排村）/ op_waterfront / op_dispersed / op_green / op_comb / op_contour / op_enclosure
                 与共用件 op_common（沿路切宅基、接回路网、门到路网的巷、补必需的公共建筑、算子适用条件 op_fits）、
                 compound 宅院成形、functions 设施、verify TP 校验与指标，内部头 plan_work.hpp 放编排状态 Work——状态只在编排里，各步拿 Work 改 Work），
                 前端 __init__（town.toml ← --set town.*、规模与地形的中英名、窗口大小）/ site（裁窗口、找聚落、上游锚点）/ style（base ← 风格 ← --set style.*、功能目录合并、
                 展平给 C++、风格哈希）/ plan（营建请求与种子）/ output / render / gallery（画廊）/ cli；随机流号 TOWN_STREAM = 22，地面噪声按来源坐标取、种子不含聚落名与风格（窗口只决定裁多大），
                 方案种子含风格（换风格地面不变）。没写 Python 参考后端（新模块，拍板 b）
config/default.toml（所有参数；[web] 段只管操作台显示，不进缓存 key；[engine] 后端开关，python 不进缓存 key、cpp 给 ①–⑨ 的 key 加 "+cpp"）slots.toml（槽位→模式/阻力档；⑧ 的 C++ 也读它）production_templates.toml（④⑤⑥模板）
config/town/town.toml（聚落营建器的工具参数：窗口、地面细化、合成地形、出图，与风格无关；`--set town.键=值`）functions.toml（通用功能目录）
config/town/styles/（base.toml 全部风格键的默认与注释，`operators = {}` 必须空着——表逐键深合并；十二个内置风格 huabei / jiangnan / huizhou / linpan / lingnan / yaodong / hakka /
                 nordic / central_eu / england / med_hill / japan；风格只写与 base 不同的键，`--set style.键=值`；模板与功能可用 ops 限定算子，[village.flat_*] 让算子要平地）
core/            C++17 核心库（PLAN-CORE；不含 Python、不含 Godot）：include/skyisle/ rng（与 numpy 逐位一致的 SeedSequence / PCG64 / 分布）、grid（栅格、噪声、连通分量、
                 形态学、倒角传播、重采样，外加 numpy 同式的 np_pow / c_pow / np_hypot / py_hypot / np_sum / np_median / pyround）、flow（填洼、D8、汇流、拓扑序）、config（扁平键值表）、
                 island/（types 群状态与各层记录、layout、territory、terrain、strat（B2 岩层）、landforms（B3 特殊的山）、coast（B4 亚格岸距）、river、build：build_terrain / build_hydro（群内各岛并行）、
                 resources / climate（含天气）/ farmland / waterworks（P6 水利）/ settle（含层级）/ generate（整群编排与粗版块降采样）；json.hpp 记录的 JSON 形）；
                 planet/（P6c：planet.hpp 行星层 ①–④ 的产物结构与 stage1–4 / run、球面与网格公共件；view.hpp 行星层 → 第三层的 PlanetView / NodeInputs；
                 P6d：graph.hpp 图算法（CSR、Dijkstra、抽样介数、弱连通分量）、civ.hpp ⑤–⑨ 的产物结构与 stage5–9 / run_society / apply_polity）；src/ 同名 .cpp（planet/stage5–9.cpp）；
                 third_party/pocketfft（np.fft 的同一份实现，numpy 2.5.2 引用的提交）；python/（nanobind：module.cpp 公共件、bind_island.cpp 第三层、bind_planet.cpp ①–④、bind_civ.cpp ⑤–⑨）；
                 tests/selftest.cpp（ctest）；tools/probe_ziggurat.py（探 numpy 的 ziggurat 表）；build.py 一键构建
```

**节点 = 岛群（R10）**：`islands`/`n_islands`/`area_km2` 等字段名沿用，语义都是「群」——一个节点 = 一个岛群 = 一个邑 = 一个水共同体；群内数十小岛属第三层，不进管线。
关键产物：`s01 planet.json`（calendar 历法、vertical 零点与垂直结构，DATA-GUIDE 第一节）→ `s03 islands.npz`（含 `area_km2`：群的总陆地 = 集雨面 = 政治体量；`territory_km2` 势力范围、`land_frac` 陆地占比、`arable_frac` 可用地率；`main_area_km2` 主岛陆地、`wall_m` 岛体墙高 = max(0, height − 300)，第三批 1；`age`/`plate` 岛龄与板块）`/plates.npz`（板块网格）`/cand_edges.npz`（无向候选边，kind 0 kNN/1 远程/2 远征/3 回退）→ `s04 wind_local.npz`（**⑤⑥⑦、check、操作台都从这里读风**：扰动后的 u/v、v_local、obstacle、wake、局部带号）`/band_local.npz`（八条局部带界 edges[8, nlon]，⑤ 的 Φ 与 ⑦ 的窗都按它）`/climate_grid.npz`（precip 由水汽模型四季各解一遍算出、q、uplift、conv、`precip_share[季]` 各季份额、`lift` 抬过逆温层的比例、`season_shift[季]` 带界位移）`/climate_islands.npz`（含 `catch` = 可用地率×陆地×降水，集雨容量；`precip_mm` 年雨毫米（= precip_mm_ref × 相对降水，线性）、`precip_eff_mm` 有效雨、**`arable_frac_eff` 降水线之后的可耕率（⑨ 人口与岛群层已垦额度读它）**、`precip_share[季, N]`；`has_river`/`river_size` 主岛河流，默认只进九格表文本；DATA-GUIDE 第二节）→ `s05 perm.npz`（perm[E,4]、perm_no_g、f_regional、g_blocked）→ `s06 routes.npz`（有向 cost[2E]、cost_no_g、cost_m[2E,4]、flow、node_flow、betweenness_sources；前 E 条 a→b 后 E 条 b→a）→ `s07 centers.json/prehist.npz/regions.npz` → `s08 fields.npz`（reach/adopt/strength/share[T,N]、conflict_by、C、L）、`iso.npz`（iso[4,N]、local_share[S,N]）、`traits.resolved.json`
→ `s09 polity.npz`（pop、state[N]（邦 id，稀疏/孤悬为 −1）、polity[N]（含船团/部落，处处 ≥0）、kind、control、dist_cap、fief（−1 直辖 / 采邑之主节点）、realm（兼并后的本朝）、circle、capital[S]）`/polities.json`（诸邦属性、suzerain、reformer、history、fronts、openings）`/history.md` → `s10_output/`（world.json、fig、ninegrid、check）。
**地区 ≠ 邦**：⑦b 的地区只是展示分区（九格表的单位）；邦是 ⑨ 的离散政治单位。九格表 ①③④ 写邑级，⑤⑥⑧ 写邦级，⑨ 文化位置仍是连续场（铁律五不动）。

## 改代码时必须遵守

1. **改了阶段代码就把 `pipeline.STAGE_VERSIONS[k]` +1**，否则旧缓存会被当成命中。只改配置不用改版本。
   （例外：只重构、产物三 seed 逐字节不变的改动不改版本——P6c / P6d 把写产物抽成 `_write`、四点四十二删 Python 参考后端都是这样；改了会让所有旧 run 白白重算一遍。）
   **第三层同理**：改了 `core/src/island/` 或它的输入拼装 / 写产物，就把 `island.ISLAND_VERSION` +1——产物写在 `out/<run>/islands/<节点>/`，
   靠 island.json 的 `meta.stamp` 判新旧（四点四十八），不 +1 就永远命中旧目录（2026-10-01 之前只查文件在不在，09-25 的产物一直被当成命中）。
2. **原则乙**：`s07/s08/s09_polity/ninegrid/polity` 不得出现 `["height_m"]`，C++ 版 `core/src/planet/stage7–9.cpp` 不得读 Islands 的 `.height`（check 的 IL-yi 与 pytest 都有静态断言）。高度只进 s04 温度、s05 落差因子、s06 爬升成本。
   **`height_m` 的口径是主岛「台面」= 陆地高程中位数**（四点十九；④ 的岛上气温就在这个高度），不是峰高——峰由第三层按岛龄 × 面积长出来，`wall_m` 因此低估了真实山高。
   **陆地不受此限**：`area_km2`/`arable_frac` 是集雨面与人口容量（docs/02 §六），可以进社会推导——高度才是「地理决定贵贱」的禁区。`arable_frac` 刻意不从 `height_m` 推（保持这条卫生习惯）。
3. **铁律五**：文化只以 share/strength 浮点场存在。不得从 argmax 派生地区/标签，不得 flood fill。P1b（每条边的 TV 差 ≤ a + b·(λ_max·cost + max L)）是硬项。
4. **原则己**：每岛必须可达（史前扩散全覆盖）、每槽位 share 和为 1（本地行 ε>0 保证）。任何会造出孤岛的改动（采样、边集、G 阻断）都要查 `IL-ji`。
5. 区域障碍必须走 **Φ 穿越归一化**（`s05.node_phi`），不得逐边乘因子（跨带通过率会随岛数指数衰减）。SK-perm 校的是本障碍的 Σf（≈1），总通过率里还叠着邻带 Φ 域重叠与局部因子，别拿它调 s05。
6. 随机数只从 `rng.stage_rng / entity_rng` 取；列表排序后使用；不迭代 set。
7. 新增参数：写进 `config/default.toml` 对应阶段段落并给注释；操作台参数面板（`index.html` 的 `PARAM_SPEC` 或矩阵区块）按需加。
8. `slots.toml` / `traits.toml` / `production_templates.toml` 不在 `default.toml` 里，通过 `pipeline.STAGE_EXTRA_SECTIONS` 进 ⑧⑩ 的缓存 key（⑨ 政治层不读它们）。新增这类独立配置文件要同步登记，否则改了不会失效。
9. **第三层不回灌**：`skyisle_gen/island/` 只读 ①③④ 的产物，`stages/`、check、ninegrid、polity、culture 不得 import 它（`test_stages_do_not_import_island` + IS-iso）。
   岛内的湖、多盆地等「会改变故事」的情形只写进 `island.json`，不改任何场。它的随机数用 `entity_rng(seed, ISLAND_STREAM=21, "island:{node}:{部件}")`，天气另加 `weather:{year}`；
   `[island]` 段不进任何阶段的缓存 key，旧 run 没有这段时用 default.toml 的默认值。
10. **算法只有 C++ 一份**（`core/`；Python 参考后端 2026-09-30 删了，tag `python-reference-final`，四点四十二）。改 C++ 之后的回归：
   重编（`$py core/build.py --test` 连 ctest）→ 三 seed `run` + `check`（0 硬 0 软）→ 只重构的改动用 `docs/probes/diff_runs.py` 与改前的 run、改前的岛群产物逐项比（常态逐位相同）；
   有意改结果的，在 DESIGN-NOTES 记清改了什么、量了哪些数（`island batch` / `transect.py` / `koppen.py` 之类）→ 全量 pytest。
   C++ 里 numpy 同式的函数（grid.hpp 的 np_pow / np_hypot / np_sum / py_sum / npround / np_median / 分位 / 插值 / 卷积，planet.hpp 的 np_maximum / np_minimum，第三方 pocketfft 等）
   是当年为了和 Python 逐位对照写的，**新代码照旧用它们**：前端与 check 还是 numpy，同式才不会在边界上两边判得不一样；踩坑表在四点二十三 – 四点二十六
   （ucrt 的 pow(x, 2) ≠ x·x、math.hypot ≠ np.hypot、Python 内置 sum() 是 Neumaier 补偿求和、numpy 标量的 round 是 rint(x·10ⁿ)/10ⁿ、np.maximum / minimum 相等取第二个、
   heapq 的 (dist, node) 平局次序、float32 栅格与 Python 浮点相乘按 float32 算……）。
   C++ 字符串只用 ASCII，中文名在前端（decode.py）映射；资源 / 聚落 / 四季的整体拼装与 island.json 摘要在前端的一处（resource_record / resource_summary /
   set_climate / set_weather / set_settlements），改输出格式改它们。

## 当前默认值的由来（调参前先看）

- **骨架第二版（2026-09-17，PLAN-SKELETON2，DESIGN-NOTES 四点十三）**：文明核心在温带。带界仍 8/28/36/62°；
  文明中心窗 `core_lat_range` 30–42°（三 seed 中心落在 33.5–37°），经度窗 50°；G = 无风带顶 36 − δ(−1.5) = **37.5°N**，经度 = D 中央 (−10)；
  D = lon [−30, 10] × lat **[6, 62]**（取窄了会有绕行走廊）；岛密度按 `lat_density` 剖面（核心 29–44° 平台 1.0、信风带 0.25、48° 以北 0.25 → 0.05）；
  障碍 B/C = 22–30°（核心 ↔ 南方），F_N/F_S = 45–62°（`kind = "lat_band"`）；人类起源 `origin = "auto"`（北信风带）。
  谷物门槛：适宜度 × f(海面冬温)，warm [6, 12] / cold [−2, 4]；季节强度 λ 10、τ 陆 8 / 海 110 日。
  旧版：中心 14–18°N、G 22°N（δ=6 的校准史在 DESIGN-NOTES 四点八）、D lat [6, 36]、按风带给密度常数。
- 分类阈值 = 船只参数：桥 0.15 天 / 小船 1 天 / 大船 3 天（1 天 = 500 km），量的是**群与群之间**的间距（群内永远密接）。48° 以北密度 0.05–0.04、极地 0.012 才出稀疏/孤悬。
- 半衰日程：daily 5–10、trade 15–30、migrate 30–60、envoy 40–80 天。阻力：低 .05–.2 / 中 .3–.6 / 高 .7–.95。
- ε0 0.02 → ε_max 0.3（隔离度尺度 3）；k_sub = 3（第三批由 2 改，见下）；每高隔离分量 3 条本地起源特征。
- 行星：半径 6371 km（地球）、自转 24 h、**倾角 34°**（骨架第二版，旧 20°）、1 日航程 500 km → 绕行 80 日。半径只经 `days_per_rad` 影响所有边的天数。
- 历法（R1，`[s01.calendar]`，C++ 的 stage12.cpp derive_calendar）：**一年 4 季 × 3 月 × 28 太阳日 = 336 日**（骨架第二版，旧 4 × 28 = 112）→ G 型星 0.96 M☉、0.94 AU、潮汐锁定 73 Gyr；
  卫星朔望月 = 一月（28 日）。历法**不反推带界**。旧版的潮汐锁定张力随之消失（`cal_accept_tidal_tension` 改回 false）。
  `mode = orbit_to_calendar` 反向：给恒星质量 + 轨道半径推每季天数。只写 planet.json，不改任何场。
- **零点与垂直结构**（PLAN-NATURE A1，`[s01.atmosphere]`，DESIGN-NOTES 四点四十三，spec 13 第九节）：高度一律相对**零点 = 浮层基准**（气压 1 atm 的高度），不是地面、不是云带顶；
  大气标高 8 km（三 seed 的台面 5–95% 在 +0.3…+2.2 km、0.96…0.76 atm，全部 50 m…3.5 km、0.99…0.65 atm；海面 −6800 m 2.34 atm、氧分压 0.49）、云带 −6200…−5000 m、岛底不低于 −4000 m（云带顶 + 间隙 1000 m）；
  零点以上按直减率 6 °C/km，零点以下取零点处的气温（④ 的 `temp_sea` 等「海面口径」说的就是零点处，改叫零点口径、键名不动）。只写 planet.json 的 `vertical` 与 ④ 摘要的岛面气压，
  不改任何场；各字段怎么用见 `docs/DATA-GUIDE.md` 第一节。`keel_clearance_m`（300 m）是生成器内部的崖脚，不是游戏里的岛底。
- 尺度口径 `[shared.scale]`（BACKLOG 第一批拍板）：全世界陆地 25,000,000 km² / 可用地率 0.10 / 100 人/km² 可耕地 → 2.5 亿人。
- 陆地（R8）：`area = 势力范围 × f`，势力范围 = 0.866 × (mean_nn × 500 km)²（与分类共用间距），
  `f = min(0.35, f0 · (ρ/ρ_med)^α · lognormal(σ=0.5))`，α=1，f0 由 Σarea = 25M 二分反解（三 seed 均 ≈0.167）。
  Σ势力范围 ≈ 356M km²（表面 70%），故平均 f ≈ 0.070（BACKLOG 里的 0.107 用的是另一种间距口径，见 DESIGN-NOTES 四点六）。
  α=1 的含义：势力范围 ∝ 1/密度，未封顶的群陆地大致相等（一邑 ≈ 3,000 km²），密接群岛贴顶 0.35 后反而更小（≈1,100 km²）；
  四个地形类的 f 恰落在印尼 0.35 / 菲律宾 0.15 / 夏威夷 0.007 / 孤悬 0.002。
- 可用地率（R9）：`arable_frac` 均值 0.10、对数正态 σ 0.35、夹 [0.03, 0.30]，纯标量、不生成岛内地形。
  集雨容量 `catch = 可用地率 × 陆地 × 降水`（s04）。用处：s06 介数源权重、s07 适宜度（× 陆地规模^γ，**γ=0.5**）、s10 九格表 ①⑤⑧。γ=0 即退回旧式。
  γ 不能取 1：归一化 log 陆地与 log 岛密度的 std ≈0.12–0.13 且相关 ≈ −0.3（旧模型 −0.21），等权会抹平适宜度的地理结构（seed 2026 的 P7 会挂）。R8 换模型后 γ=0.5 三 seed 直接通过，没有重校。
- **政治层（第四批 R7，2026-09-11，`[s09.polity]`）**：人口 = P1 × 陆地 × `arable_frac_eff`（④ 的降水线：能种多少地）× 单产（有效雨 250 → 700 mm 从 0.6 到 1，A3 起；≈1.7 亿，旧 clip(降水/0.25, 0.25, 1) ≈2.3 亿）；控制权重 w = 商旅成本 × (短渡内 1 / 群间飞行 **3**) + R0·L_trade，
  控制力 exp(−w/R)，**R0 = 2.2 天**、θ = 0.2、R 随核心实力^0.25 放大（夹 0.5–2.5）；核心实力 S = 控制范围内 Σ 人口 × 控制力，**立都门槛 = S 中位（`capital_min_strength_frac=1.0`）**，
  0.5 时邦数翻倍、中位只 5 邑。三 seed（骨架第二版）：591/590/607 邦，中位 9–10 邑，密接之都的邦是中疏之都的 3.5–4.8 倍（P8 阈 1.5）。
  宗主半径 = R0 × 0.5 且不随实力放大、王畿锁定（否则中心处人口最稠，宗主反成圈内最大邦）。变法 80 年前、用 20 年、动员 ×3、守方 ×2、宗主顾忌 ×3；
  兼并只由变法之国发动（docs/02 §八 ↔ 铁律三 的调和），先易后难，占领消化按 docs/04 §四 的 [3, 8, 15, 25, 60] 年。来由见 DESIGN-NOTES 四点九。
- **seed 2026 是 P7 的哨兵种子**（拒绝点数只有 seed 42/7 的零头；`k_sub=3` 与 ⑦ 的 `secondary_per_circle=3` 都是为它定的：它的 NE 圈分不到全局峰值）。改 ⑦ 适宜度或次级起源相关的东西，先拿它试。
- **seed 7 是 P6 的哨兵种子**：G 邻域只有 8–13 个岛，枢纽 3–4 个，其中两个是穿 D 干线的门户型（反事实里流量反而上升）。改骨架、板块、绕道弧的东西先拿它试。
- **④ 四季、降水线、⑦ 驼峰（PLAN-NATURE A2–A4，2026-09-30，DESIGN-NOTES 四点四十四）**：四季各解一遍水汽，带界整体摆 `season_band_shift_k` 0.35 × 倾角 × A_海（±5.2°）；
  云带是逆温层：蒸发源 × 抬升率（基础 0.1、辐合 0.4、深对流 0.5（季温 24 → 28 °C）、风暴 **3**、撞墙 0.8）× exp(2 × min(辐合, 0))，ε 另乘 exp(0.8 × min(辐合, 0))；
  岛群季风 **0.8** × 陆地覆盖（抹开 4°）× cos(季相 − **陆**的滞后)；下沉 **1.5 / 0.6**（A5 由 A2–A4 的 1.2 与 2.0 / 0.8 回调：岛群四季里看是华北式的冬天无雨）；
  年雨 = 四季水量平均按 P98 归一。三 seed：陆地 < 400 mm 24–25%、< 200 mm 8–11%，
  28–36° 岛密处 460–580 mm（夏半年 68–72%）、岛稀处 250–340，西风带 48–56° 1,050–1,240，赤道 3,200；毫米 = 4000 × 相对降水（线性）；
  降水线：有效雨（休眠季的雨算一半，生长季按零点口径季温 5 → 12 °C）250 / 400 / 600 mm → 可耕率 × 0.15 / 0.5 / 1；
  ⑦ 驼峰 [500, 1000] mm、宽 0.5（九个中心八个在 580–980 mm）。调参只看剖面，没对着地球比例凑。
- 第三批（2026-09-10）的默认值：板块 `[s03.plates]`（汇聚 ×3、离散 ×0.15、叠层核阈 0.9、D/G 邻域不修饰、乘子归一）、
  障碍增益 `obstacle_gain=4`、带界位移夹 ±3.5°、水汽 `evap_temp_coeff=0.03`/`precip_conv_k=0.6`/τ 4 天、`k_sub=3`。来由见 DESIGN-NOTES 四点八。
- 验收阈值中几个是按三 seed 校准过的：P5 强度比 2.0、重心顺风占比 0.5（起源风速门槛 2 m/s；A5 由 0.75 放宽，四点四十五）；C5 中心周边岛上全年温差 ≥ 20 °C、冬温 ≤ 5、夏温 ≥ 20；
  「干旱」口径 `arid_mm` 400 / 「湿润」`humid_mm` 800（A3 起按毫米）、C3 干岛占比 [0.15, 0.40]；河流 `river_min_mm` 400（A5）；
  P5 风稳的起源不到 `p5_min_origins` 4 个时只按强度比判（A4），重心顺风占比 0.75 → **0.5**（A5：只查不反号，强弱归强度比）；P6 混合度 0.3 + 坍缩占比 0.25（相对分位只报告）；P7 reach≥0.3、伴随器物≥0.4、地区覆盖 0.2；P3 用聚束障碍分比值 ≥1.5（全局秩相关只参考）。

- **岛群生成器（2026-09-17，PLAN-ISLAND，`[island]`）**：栅格 100 m（群外框 > 2048 格自动加倍，38,000 km² 的最大群落到 400 m）；岛数 12–80、Zipf 1.1、最小岛 0.3 km²；
  岸距 1–15 km（beta(1.3, 2.2)）、**没有索桥**（P5 用户定，四点三十六；09-27 先砍到 0.1 km / 30 m，旧 2 km / 250 m），群内全是短渡；**高度**：台面 = height_m（陆地中位），起伏按岛龄对数插值（1000 km² 时新岛 3000 → 老岛 450 m）× (面积/1000)^0.3 × 对数正态 0.25，
  中位分位 新 0.22 / 中 0.2 / 老 0.38（台面离岛底太近时先压到 0.12 再压起伏；台面 < 600 m 的岛底 = 0.5 × 台面）；#1165：岸缘 570 → 峰 1,830 m，坡中位 5°、>15° 占 12%；
  下切 15 / 8 轮在原生分辨率上（B1：≤ 2048 格，旧 ≤ 320 格的粗网格；每轮全岛填洼，旧的 40 次迭代只填到离岸 4 km；carve_k 0.3、m 0.5、河道阈 0.3 km²、随机流向 p 1.5）；湖 = 填平深 ≥ 3 m 且 ≥ 0.5 km²（`pit_keep_m=2` 让它少见）；
  常年河按流量：年均 ≥ 0.3 m³/s（流量按 Budyko 径流，A5：500 mm 的温带群径流系数约 0.2–0.3、700 mm 约 0.35–0.4，高处比低处大一倍；不够则 0.2 × 主岛最大汇流；旧固定 25 km²，四点二十），河宽 / 水深真实比例（C1，旧 ×8 / ×3）、干流下切 25 m、
  河床最小比降 1 m/km、谷底宽 50·A^0.4 × 岩性 [1, 1.6, 0.6, 1.3, 0.5] × 岛龄 [0.5 → 1.6] × 比降系数（C2，旧的漫滩 = 10 × 河宽作废）、河谷最远 2 km；
  集水核 `[island.water]`（C4 / C5，四点四十七，都是我定的、等用户看）：core_gain 1、强度 cbrt(km³)/10 夹 1.5、湿度 (P/1500)^1.5、迎风 ±0.5；云雾林 ≥ 150 mm 且 ≥ 1/4 局地雨；
  热的 0.001 进岩体、每 5 MW 一处核山温泉（大核山 ≥ 0.25）；基流比例 [0.55, 0.25, 0.4, 0.2, 0]、退水 [60, 15, 30, 15, 30] 天；
  **水位面（B+A，四点四十九，都是我定的、等用户看）**：`gw_r_over_t` 8e-5 /m（水位丘 ≈ 这个值 × 到排水口距离² / 4：越小越平；#1165 埋深中位 21 m、p90 146 m、最深 488 m）、
  含水层最小厚 5 m、粗格 4（SOR ω 1.85、300 轮 / 0.05 m 收敛）、打破平局的朝岸微倾 1 m/km（最多算到 2 km，只为定流向）；
  泉线 1 km 一段、≥ `springline_min_ls` 0.5 L/s 才记；**泉 / 崖瀑**要出水 ≥ 本岛出口段的 `spring_pct` 0.85 / `spring_fall_pct` 0.97 **且**含水层厚度 ≥ `spring_min_aquifer_m` 20 m，
  其余算弥散渗出（旧键 `springline_fall_ls` 绝对 5 L/s 作废）；#1165 629 弥散渗出 / 90 泉 / 41 崖瀑（直径 100 m、21 岛，generate 7.0 → 7.6 s）
  河的数据 `[island.rivers]`（C3，四点五十：`w_bf_m` / `d_bf_m` 是真**平岸**（= 年均口径 × (q_bf/q_mean)^站内指数 0.26 / 0.40 ≈ ×1.5–2）、`w_mean_m` / `d_mean_m` 是年均流量下的水面；terrain.npz 的 w_ch_m / d_ch_m 是**河道**（× bf_ratio_channel^站内指数））：砾床 Shields 0.05 / 砂床 1、BQART 岩性 [1, 1.75, 0.6, 1.5, 1]、全垦 × 2 全林 × 0.7、瀑布合起来 ≥ 20 m 且平均比降 ≥ 0.25、雪 ≤ 0.5 °C、度日 3 mm；
  地形区的局地起伏按 4 km 方窗，山地 ≥ 300 m、丘陵 ≥ 100 m，外加相对高度判据（旧的平岛上只按起伏判一格山地都没有；有真实起伏后主要靠起伏判）；土层到 60° 才归零、坡向湿度修正 ±0.2、成林土层门槛 0.2（按平岛定的 40° / ±0.4 / 0.3 在真实坡上是一片灌丛）；盆地 = 汇流 ≥ max(5 km², 2%) 的河口集水区，≥ 15% 岛面积算大盆地；
  相对降水 → mm：④ 的 `precip_mm_ref` 4000 × p（线性，A3；旧 150 + 3850 × p^1.3 作废）、已垦额度读 ④ 的 `arable_frac_eff`；四季降水按 ④ 的份额、带界摆动读 ④ 的每季位移（A5，±5.2°）；季型阈值：四季分明 ≥ 20 °C、冷暖两季 ≥ 8 °C、雨旱 2.5 倍、风暴 / 窗口季差 0.25；
  雨日比例 0.12 + 0.30 × (季雨量/1000)^0.7、湿→湿持续 0.45、伽马形状 0.8、风暴日比例 0.35 × 强度^1.2（布尔覆盖率反解事件数）、云海漫顶只在峰高 < 800 m 的群。
  IS-daily 用 600 年样本（四点十四定 60 年：30 年时单季标准误约 5%；A5 起加长，干端的年雨偏态重，60 年的均值常低一成、样本标准误也偏小，2,000 年回到 1% 以内，四点四十五）。SET-home / SET-water 在没有村的群（荒漠里只有散户）不要求邑治旁的候选、不查村的水源。
  聚落 `[island.settle]`：5 人/户、村 8–80 户（邑治田块可到 300）、村址离田 ≤ 800 m、村间距 1 km（软）、蓄水池只给 ≥ 5% 岛面积的盆地、
  都的城居率 0.3 / 集聚率 0.05（变法之国 0.10）、郭 5 km（同岛或岸距 ≤ 3 km 的岛）；非农 10%、泊场（船台）6 格内坡 ≤ 4°、开垦半径 1.5 km × √(户/40)（1 km 时林地只降到 74%）；
  村取石 5 km（石料按露头算之后林茂的低地 3 km 内常常没有能开的石头；飞船运石）、取土 / 砂 2 km，1.5 km 内已有采场就合用。
  **有的地方就是没人**（P5，2026-09-29，四点三十六）：宜垦 = 坡 < 25°、最暖的月份（年均 + 半个季节温差）≥ 10 °C、土层 ≥ 0.2、湿度 ≥ 0.25 或近水（剖面八群占陆地 81–91%）；
  已垦 = 按适宜度好地先占、填到额度（与上等地同格数），宜垦片（8 连通、不跨岛）头一轮分到的地够 20 户、有水（1 km 内河湖溪泉或年降水 ≥ 400 mm）、能落船才常住，
  不够的整片让出来；远近不排先后（试过跨岛 × 0.95 × exp(−岸距 / 50 km)：三群只剩主岛有村，太过）；撂荒 = 额度外贴着田的 8%（1–25 年）+ 废村（每群 ≤ 3，撤空 5–60 年）；
  没有农户的岛上的浮石采石村 / 矿村 / 窑村是工棚，烧炭营季节住；放牧 ≤ 4、庙 50%、墓岛 35%（烽火台 ≤ 2 P7 起归中转站）。剖面八群 < 30 km² 的小岛常住的 25–100% → 0–62%（多数 0–9%）。
  **大岛保底**（用户 09-29 定）：主岛以外 ≥ 30 km²、有合门槛的宜垦片的岛先分一块连成片的 20 户的地（一个村，户从主岛匀过去）；村多大按门槛定（试过 80 户：主岛占比 72.9% → 68.7%，
  且 3 座岛长不出连成一块的 80 户的地反而保不住）。剖面八群 ≥ 30 km² 的非主岛 107 座全有村，住在 < 30 km² 小岛上的人 2.0% → 1.1%（人没少，住处变了）。
  **水利**（P6，2026-09-30，四点三十七，`[island.works]`，都是我定的、等用户看）：渠首在汇水 ≥ 10 km² 的常年河或 ≥ 30 km² 的季节性溪涧（配堰塘）上、从高往低挑、隔 3 km；
  渠比降 0.5 m/km、挖穿 3 m、走出 6 km 的工、横坡费工 100、田里的格点 0.8 km、新接的渠 ≤ 0.5 km + 3 km × 田的面积才修（第二版没有这条时每 km² 灌田配 3.7–7.2 km 渠，加了以后 2.5–3.4）；
  村塘 40 m²/户、山塘 / 堰塘 = 所灌田的 3%；圩田：湿地片 ≥ 0.3 km²、周围 2 km 的平地（坡 < 5°）第一遍种了过半才排干、600 m 一圩、圩塘 5%、水田线 800 mm；
  **圩田在额度之内**（第二遍先占、挤掉最外一层的地）。剖面八群圩田 0–38 km²（#5498 湿地的 19%、#6329 2.8 km²、#6610 26 km² 全是泽田），谷口的渠 92–489 km、灌田占已垦 7–30%。
  **有水利就有人维护、原始地貌与人工地貌分开记**（P6b，2026-09-30，四点三十九；用户定 L30 / L31，下面的数是我定的、等用户看）：管水利的村走得到 `manage_walk_km` 2 km（直线，用户建议的走路半小时）；
  圩田按 `polder_village_blocks` 3 × 3 圩一组成田（1.8 km 见方），整组走得到就挂村、走不到就落圩村；渠首一村一堰、隔 `village_head_sep_km` 1.5 km，灌区按格算（旧键 `canal_head_sep_km`、`canal_cmd_frac` 作废）；
  镇只从 ≥ `[island.market] town_min_village_hh` 8 户的村里挑。剖面八群：水利离管它的村全在 2 km 内（改前离最近的村两三成超过、渠首最远 7.3 km），渠首 12–47 处、谷口的渠 30–160 km，
  灌田按格 10–48 km²（占已垦 2–8%；按 P6 的整块口径 33–121 km²），圩村 33 个、挂圩田的村 67 个，废塘 13 口、废渠首 0（废村都在没有大河的小岛上）。
  **水利分级**（2026-09-30，四点四十，用户定 L32：岛群层只出都江堰级的；L33 次序照我的建议；下面的数是我定的、等用户看）：`[island.works] village_works = false`（村级的归营建器）、
  大堰 `big_max` 2、汇水 ≥ `big_src_min_km2` 30 km²、抬水 2 m、干渠 0.3 m/km 走出 25 km、灌区坡 < 5°、年灌水 600 mm、灌得到 ≥ 20 km² 且 ≥ 额度的 10%、适宜度 × 2、支渠格点 1.5 km；
  settle 里大半在大堰灌区的连通块不按邑治 300 户切。剖面八群五群有大堰、灌已垦 12–69%，住在主岛 70.9% → 72.3%，开局 #6329 的邑治回到主岛。
  **镇、邑治、航船、中转站**（P7，2026-09-30，四点三十八，`[island.market]`，都是我定的、等用户看；前提 L24 用户定：一般人家没有船，本岛走路、跨岛搭航船）：
  大泊场 = 坡 ≤ 4°、没在种没撂荒的空平地（林子算）开一圈、1 km 见方窗里够 0.5 km²、隔 5 km，船数 = 平地 × 0.5 / 2000 m²（满窗 302 条）；村的船台照旧（封顶 10 条）；
  挑镇：本岛 6 km 走路 + 跨岛航船船程 ≤ 40 km（风的系数顺 0.7 / 逆 1.5 × 年均风的稳定度），挨着 2 km 内大泊场的 × (1 + 0.3 × 船数 / 150)，还没被服务的户 ≥ 150 才挑、镇距 10 km；
  归镇：本岛 10 km 内走路，再远或跨岛搭航船；航船线至多 6 个村、60 km、三日一班（≥ 400 户每日）；邑治 = (服务户 + 2 × 航船送来的户) × (1 + 泊场船数 / 150，封顶翻倍) 最大的镇；
  中转站：⑥ 邻边来回流量 ≥ 100 成口子（方位差 < 60° 并、至多 4 个），落在方位 ±60° 内朝外最远、有平地能存水、看得远的岛上（没有就主岛崖边），关卡 / 过夜（进来一程 ≥ 0.4 天）/
  候风（出去一程成本 ≥ 1.5 × 天数）/ 避风（风暴 ≥ 0.15）/ 换船（枢纽群流量最大的口子），户按角色给、至多剩下非农户的 25%；烽火台 ≤ 2、隔 90°、轮班 0 户。
  剖面八群：「有别的岛上的人来赶集却没航船」的镇每群 2–10 → 0，镇离大泊场中位 2.7–13 → 1.4–3.7 km，航船 8–21 条，中转站 1–4 处（常住 0–51 户），八群里七群的邑治换了。
  资源 `[island.resources]`（四点二十一）：赋存区阈值 矿 / 硫磺 / 砂金 0.15、黏土 / 砂砾 0.3、石料 0.4，隔一格的碎块算一处、≥ 0.05 km²（不合并时 #1165 有 1,940 区，合并后 736）；
  岩类可放区的丘陵坡门槛 8°（「严格按地表、林地一律不放」在林多的群只剩 1–5% 的地、九成以上的村取不到石头，用户选了按地形判）；
  **可放区 ≠ 石料区**：石料按露头算，坡 12° 起算、30° 满，裸岩 / 高山草甸 / 峡谷壁至少 0.6，林坡 × 0.8，> 40° 算崖面（第一版 8° 林坡就算，#1165 石料区 31%、看上去全岛是石头）；砂金增益 8。
  **来历**（P3，2026-09-29，四点三十四）：B2 起岩性读岩层（石料赋存区按岩性类分开成区；旧的剥蚀指数分三层作废），剖面八群的石料面积 石灰岩 68–95% / 辉长岩 5–32%（老岛才多辉长岩、蛇纹岩）；盐丘每 100 km² 沉积盖层 0.05 个（约 2 km²，八群里五群有、0–3 个），盐井村 20 户/km² × 品位、≤ 80 户；贝壳化石每 km² 海相石灰岩 0.1 个；
  骨架空洞每 km² 浮石露头 0.15 个（与原来熔岩管的个数相当）；温泉只在新岛（八群 67 → 10 处，#2051 8 → 0）。
  季型判定的分数 = 差异 / 该项门槛（温度按冷暖两季的 8 °C，不是 20），取 ≥ 1 的最大者；三 seed 全量：四季分明 57–60%、冷暖两季 19–23%、风暴季 18%、雨旱季 1.5%、常夏 1%，
  西风带以北 100% 四季分明，信风带一半冷暖两季一半风暴季。
  势力范围 `[island.territory]`（四点二十二）：缝 3 km（两边各退 1.5）、看群心距 ≤ 3 × 等效半径之和 + 40 km 的邻群、主岛转 8 个走向、拉长 1.6 / 2.4、重摆时群内最小岸距 0.3 km；
  seed 42 全行星重摆 67%（密接几乎全部、中疏 68%），仍越过分界线的 34 群（③ 里群心挤在一起的，第三层摆不开）；IS-terr 硬 / IS-terr-gap 软。
  **地貌 P4（2026-09-29，四点三十五，Zhouzhu PLAN-LAND P4）**：新岛是拱（`median_frac_arch` 0.35、最低 0.3，旧 `median_frac_young` / `cone_*` 作废）；
  多核嵌合 `multicore_frac` 0.3（**用户定三成**：汇聚带边界核 ≥ 0.5 的大主岛的 30%，折全行星 ≥ 300 km² 主岛 17%）、≥ 300 km²、不是新岛，缝脊 0.12、浅槽 0.2 × 0.5 等效半径，根深系数 5；
  谷收拢 `[island.hydro] capture_reach_km` 3（**用户定「中」**：弱 1.5 / 强 5；山在中间的岛汇水 ≥ 5 km² 出岸缘的河少两三成）、大谷数 = 面积 / 150 km²、收完 ≤ 岛面积两成、新岛 × 0.1；
  局地雨按本岛的起伏（**用户 09-29 定**：`oro_rise_per_km` 0.4 × min(高出本岛岸缘 km, `oro_rise_max_km` 2)，旧 `oro_per_km` / `oro_lo_km` / `oro_hi_km` 按主岛台面、作废）、
  `windward_gain` 0.3、迎风在 3 km 块均值的起伏上量；湿地 `wet_*` 改口径（旧 `wet_slope_deg` / `wet_acc_km2` / `wet_precip_mm` 作废），
  指数 ≥ 5e5、周围最陡坡 < 2.5°、离崖缘 2 → 6 km、雨的三次方。
  浮高 `[island.float]`（2026-09-27，四点二十八，Zhouzhu 浮高计划 G 期）：主岛不动，其余岛整座平移 δ——z = 0.65 − 0.75 × Δ岛龄 / 0.08 + N(0, 1)，
  往上 1500 × tanh(550 z / 1500)、往下 500 × tanh(450 z / 500)，往下的再按 (岸缘 − 20) / 500 缩（岸缘 ≥ 20 m、不在下限堆一摞）；
  三 seed 全部非主岛：往上 70%、|δ| 中位 405 m、p90 1.02 km、最高 +1.47 km、与 Δ岛龄秩相关 −0.58；索桥当时按平移后的岸缘高差判（seed 42 全行星：旧判据 75,768 → 27,513，0.1 km / 30 m 下 64 座；P5 起索桥整个去掉）。
- **生成器 `[engine]`（2026-09-27，行星计划 P6a – P6d，四点二十三 – 四点二十六；四点四十二删了 Python 参考后端与 `backend` 键）**：
  `threads = 4`（群内各岛并行、⑥ 抽样介数按源并行，结果与线程数无关；`island lod` / `floats` 多进程时自动 1）。island.json 的 meta 有 `"engine": "cpp"`。下面是移植时两个后端的实测，留作用时的参照：
  P6b 实测：seed 42 / 7 / 2026 共 210 群（对照 30 + 随机 180）整套产物逐字节相同；整群 generate（不写产物）4 线程中位 0.62 s、最长 1.65 s（python 7.3 / 22 s）；
  全行星粗版 10 进程约 12.5 分钟（python 28 进程约两小时）；带天气的实跑（四点二十七）：2 km + 1 km 22 进程 8.5 分钟，每群中位 1.07 s，与 P0 的粗版逐位相同。
  P6c 实测：三 seed 的 ①–④ 产物两个后端逐位相同（⑤–⑨ 与 ⑩ 的数据随之相同，图只差标题里的 run 名），三 seed 的 check 在 cpp 下 0 硬 0 软 0 报警；
  ①–④ 用时（不写 npz，中位）python 0.77 s → C++ 0.39 s（单线程；numpy 本来就是向量化的，大头是 ③ 的 kNN 与 ④ 的水汽推进）。
  P6d 实测：三 seed 的 ⑤–⑨ 两个后端逐位相同（①–⑨ 全部产物与摘要、⑩ 的 json / 九格表；图只差标题里的 run 名），python 后端重跑与改前逐位相同、key 不变；
  ⑤–⑨ 用时（不写产物）python 46 s → C++ 0.77 s（⑥ 的介数 29 s → 0.1 s）；整条管线 ①–⑨（写产物）python 43 s → C++ 2.5 s；`planet_run` ①–⑨ 在内存里 1.2 s。
  切默认后 seed42 不变：15 个群（含 2051 / 2050）cpp 在 out/cpp-seed42 上生成的与 python 当场重生成的只差预览图标题里的 run 名。
  删 Python（四点四十二）：三 seed 全命中缓存、seed 42 从 ① 重算与原来逐位相同，五个剖面群的产物与删之前逐字节相同。

## 未做 / 可改进（按价值排序）

- **自然层第二版（`docs/PLAN-NATURE.md`，2026-09-30 用户逐条定；A 开工：A1 零点与垂直结构（四点四十三）、A2–A4 ④ 按季节算 + 毫米换算与降水线 + ⑦ 驼峰（四点四十四）、A5 岛群层接新 ④ + Budyko 径流（四点四十五）已做，A6 验收报告给用户看；
  A 的遗留：P5 的重心判据是放宽的（该改成与同一片岛不看风的重心比）；荒漠群只有散户与蓄水池（C 的集水核跟着湿度走，干的群照样干，没长出绿洲）；航线仍按年均风。
  B（四点四十六，10-01）已做：B0 河宽不夸张、B1 原生分辨率侵蚀、B2 岩层、B3 特殊的山、B4 亚格岸距；B 的遗留：切出 V 形谷之后没有平的谷底，湿地少三到七成、圩田跟着少、邑级大堰 5 群 → 0 群，
  等 C2 谷底宽补；方山、蛇纹岩只在老岛（全行星 0.8% 的群）；整群 generate 中位 0.62 → 1.51 s、全行星粗版估 15 分钟）。
  C（四点四十七，10-01）已做：C1 河道格 height = 平岸水面、窄河不记成水面，C2 谷底宽（比降系数限住峡谷），C3 rivers.json 的分段 / 沿程每点 / 瀑布 / 逐日径流指数，C4 集水核（凝结水只进水账、云雾林、核山温泉），
  C5 地下水（基流、崖壁泉线、泉按岩性），C6 IS-water / IS-valley；谷底 0 → 28–182 km²；C 的遗留：湿地没回来（谷底边上的阶地坎，六群又少三到七成）、邑级大堰仍 0 群（#5498 差门槛一点，门槛是四点四十定的，等用户看）、
  平岸宽深仍按年均流量（干而暴的群偏窄）、逐日径流指数只一年；generate 中位 1.64 s）**：设定在 `docs/spec/13-浮石与陆地的生灭.md` 第二版（第二个超自然要素**集水核**、浮力随高度减弱得快、垂直结构与厚大气、闭孔两层、特殊的山）；阶段 A 行星层 ④ 按季节算（副高下沉、逆温层之上默认干、弱的岛群季风）+ 降水线 + ⑦ 驼峰 + 零点写进 planet.json，B 侵蚀细化 + 岩层进侵蚀 + 特殊的山 + 亚格岸线，C 河去夸张 + 漫滩照地球 + 河的数据 + 集水核 + 地下水，D 游戏；开工前的 numpy 环境与 ME Pro 验证（四点四十一）、删 Python 的算法（四点四十二）09-30 都办了，接着开工 A。用户的总原则：原理定了以合理为主，要硬凑才对得上就改原理；地球的数只当参照。

- C++ 核心库（PLAN-CORE 第九节）：①–⑨ 与第三层都已移完，Linux（ME Pro）09-30 编过并验过（四点四十一），Python 的算法 09-30 删了（四点四十二）；
  **⑨ 附庸判定的怪处**（`cap_dist[t].get(capitals[s])` 拿都城节点号查以邦号为键的表，三 seed 的附庸只有 0 / 1 / 0 个，修正后 147 / 158 / 164 个，兼并史不变；四点二十六）C++ 里照旧，修不修待定；
  游戏（P6e）要一份不经 Python 的配置来源（C++ 的 Config 现在由前端展平 default.toml 给）

- Monte Carlo 引擎（`s08.engine="mc"` 只留接口）、软先到权重（`first_arrival_weight` 默认关未实现）
- 季节窗口进模型（现只有 `seasonal` 标志与 ④ 的窗口比例；岛群生成器已给出每季窗口，但管线不读它）；政治性障碍只支持经纬矩形覆盖
- 水利（P6）：渠只是线、不改地表也不回头改墒情 / 适宜度（「渠和塘算近水」是游戏的事）；平地上的渠是 45° 与正南北的折线；纵浦横塘是正南北东西的网格；没有「长藤结瓜」、北方的井；
  挑圩田只看第一遍的压力；粗版不跑聚落、看不到渠与圩田（DESIGN-NOTES 四点三十七末尾）。P6b：几个村合用一条长渠（渠长轮值、按田出工）没做，渠首多贴着管它的村 2 km 的边；
  废渠首 / 废渠只在测试里走到（剖面八群与 30 群的废村都在没有大河的小岛上）、废圩没做；小圩村（< 8 户）照样是村（不当镇）；P7 挑镇、挑邑治对户数敏感，#6329 的邑治跟着换到岛 2（四点三十九末尾）
- 水利分级（四点四十）：营建器只记下、画出邑的渠与本村的分水口，营建算法不避渠、村里的水不从分水口接；村域（村外田里的渠、塘）窗口没做；一个水系至多一处大堰；没有大堰的群岛群层只剩圩区（打井、溪涧的小渠归营建器）；灌区顺等高线成带，个别村的田拉成长条（#6329 村066 的田 8 km）。
- 镇与航船（P7）：大泊场只是普查（点与面积，不从地表划出来）；航船线是村心之间的直线、按年均风，没有按季 / 按日的风与时刻表；中转站看 ⑥ 的长途流量，⑥ 的背水群（#2051）没有关卡；
  站址不看平地够不够大；镇的市户没有细分行当（船行、浮石行、修船铺……是游戏的事）；邑治可能是小村长出来的镇、也可能不在主岛（#5766）；游戏的 C++ 路要从行星包给 `NodeInputs.routes`（DESIGN-NOTES 四点三十八末尾）
- 岛群生成器：资源不进村址选择（村定了再去找石 / 土，矿镇落在矿旁）、赋存区的储量只有面积 × 品位没有吨位；岩性 B2 起有栅格（terrain.npz 的 lith）、
  **水位面 B+A 起有**（wt_m / wt_depth_m，四点四十九）；逐日径流指数只一年；**聚落层还没读埋深**（井 / 取水点 / 定居门槛要水，下一轮改）；
  泉的等第只看「本岛出水分的分位 + 含水层厚度」，没看岩性与岩层 / 浮石交界在崖壁上出露不出露；水位面是粗格解（排水口在细格上钉回地表，其余格是粗格插值）；
  **河道宽用假设的平岸 / 年均比 5.0**（地形阶段还没有逐日天气，四点五十一）：真实温带 3.8–4.6、本模型的日指数年最大 4.8–14.9（**偏急一倍**，天气模型的事，没动）；drainage density 0.56 km/km²（真实温带 1–2，`stream_min_km2` 1.0 偏保守，没动）；
  河宽 B0 起是真实比例、四点五十起平岸 / 年均两个口径分开，不是水文模型；**四点五十一 按文献重做**：那两个「标定点」是**平岸**口径（泰晤士 60 m = 年均水面 40 × 4.3^0.26），`a = 5.0` 配年均流量是对的；错的是四点五十 用下游指数 0.5 换算站内（多放大 1.6 倍）——已改 `ratio^0.26`，且**切出来的河道改成平岸河道**（`bf_ratio_channel` 5.0；栅格键改 w_ch_m / d_ch_m）；默认视图的河道线宽是示意（有最小像素下限），控制台有「按真实比例」开关；不做岛内逐日空间分布；老岛台地的宽谷偏少；可耕地偏向沿河带；散户多（已垦按格排先后，田是一片片的）；粗版不跑聚落、地表里的田仍按上等地画；`island batch` 的三 seed 统计见 DESIGN-NOTES 四点十四。
  **版本戳只管控制台**（`products_stale`）：`transect.py` / `works_view.py` / `market_view.py` / `koppen.py` 直接读 `islands/<节点>/`，不看戳——量之前先确认那批产物是当前的（或 `island <节点>` 重生成一次）
- 九格表 ④ 特有种 / ⑦ 外观 标【待填】；⑨ 文本量词与归因还比较模板化。⑧ 的世仇 = 接壤且势均力敌、界边最多的邻邦（启发式）
- 政治层：兼并史是静态快照 + 逐邦顺序（无年内事件、无分裂/复国）；附庸只一层；邦名是 `邦NNN` 占位；南圈与 NW 圈不发生变法（docs/11 §八）
- 操作台：路径计算需服务端（单文件版不可用）；边层默认只画流量前 N；无撤销/对比两个 run 的差分视图；气象层没有粒子动画（流线虚线已够用）；水汽 q / 抬升 uplift 已在网格通道里
- 操作台改前端时注意：globe.gl 会清空 `#globe` 的内容，遮罩/悬停框必须放在 `#globeWrap`；`pathsData` 的点高度靠 `pathPointAlt(p=>p[2])` 才生效（DESIGN-NOTES 五）
- 操作台左栏按「视角」组织（地理/气候/航运/文化/政治，`index.html` 的 `LENS`）：新控件或新下拉选项要加 `data-lens="…"` 标明属于哪些视角，否则所有视角都显示；
  新图层开关要登记进 `LAYER_IDS` 并写进相应视角的 `on` 预设。「模式」全局一个（`modeSel`），边相关处用 `edgeModeVal()`（「全部」按商旅）。右侧「开发」标签 = 验收 + 参数。
  地址栏 `#lens=climate&run=seed42&node=1165` 是页面状态：`selectIsland` / `setLens` 都会 `writeHash()`，启动时先读再 `setLens`（否则被重写掉），有 node 就选中并飞过去；顶栏「节点#」回车跳转；岛群调试台的返回链接带 node 回来。
- 性能：8000 岛全跑约 2 分钟（s06 介数 30 s、s10 图 50 s 为大头；s09 政治层约 3 s）；20000 岛未系统测试
- `check --seeds` 多种子批跑、`viz diff`、`config diff` 未做
