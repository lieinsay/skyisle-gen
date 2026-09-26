# 空岛行星生成器 —— 工作约定（Claude 上下文）

这是 `docs/spec/12-扩散模型.md` 的实现：一颗浮空岛行星的地形、气候与文明生成器。
包名 `skyisle_gen`，命令 `skyisle`。**先读本文件，再读 `docs/DESIGN-NOTES.md`（决策与踩坑全记录）。**
上游规格随仓库带了两份快照：`docs/spec/12-扩散模型.md`（规格书）、`docs/spec/11-世界总图.md`（骨架定稿）。
其余上游文档（`01-设计铁律` 硬约束、`02-世界与地理` §3–5、`08-地区设计规程` 九格表格式、`04-社会与变迁` §3 四模式）
留在原 Zhouzhu 设计仓里，本仓库不含副本；下文与 `docs/` 里凡写 `docs/0X-…` 的，都指那边的文档。

## 环境与命令

- Python 3.12：`%LOCALAPPDATA%\Programs\Python\Python312\python.exe`（不在 PATH；PowerShell 里用 `$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"`）。依赖仅 numpy + matplotlib（+ pytest）。**不引入 scipy/networkx/pandas。**
- **ME Pro（Debian，无显示器）**：⚠️ 那边是按旧名字装的（venv `~/.venvs/zhouzhu`、软链 `~/.local/bin/zhouzhu`、`~/.bashrc` 的 ZHOUZHU-DEV-ENV 段），
  改名后**需要重装一遍**（venv 里装的是可编辑包，入口脚本名变了）。该段里已 `export MPLBACKEND=Agg`。中文图标需 `fonts-noto-cjk`（已装，字体回退表里列了 Linux 三个名字）。
  `pipeline` 与 `serve` 不要同时跑；操作台绝不绑 `0.0.0.0`。
- 一律在仓库根下执行：
  ```
  $py -m skyisle_gen.cli run --seed 42            # 十步全跑（约 2 分钟；只改 [s08] 约 15 s；只改 [s09.polity] 约 1 分钟，大头是 ⑩ 的图）
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
  $py -m skyisle_gen.cli island stats --run out/seed42   # 全量季型统计（只算气候，8000 群 4 s）→ islands/season_stats.json；操作台气候视角「季型」着色读它
  $py -m skyisle_gen.cli serve                    # 3D 操作台 http://127.0.0.1:8642/（完全离线）；岛群调试台 /island.html?run=seed42&node=1165
  skyisle serve --host 192.168.0.116,10.8.0.12 --no-open   # ME Pro 上这样起（--host 可多地址；拒绝 0.0.0.0）
  $py -m skyisle_gen.cli viz web --run out/seed42 # 单文件 viewer.html（内嵌 globe.gl）
  $py -m pytest tests -q                          # 48 个测试，约 30 s（tests/test_island.py 跑一个 1600 岛的小世界到 ④）
  ```
- 验收基线：**seed 42 / 7 / 2026 三个种子 `check` 必须全过（0 硬项 0 软项）**，改动核心公式或默认参数后都要重跑这三个。
- PowerShell 向 `python -c` 传含引号的代码会被破坏：写成脚本文件再跑。
- 产物目录 `out/` 已 gitignore；`config.resolved.toml` 是 `check/viz/probe` 读取配置的来源——改了 `[check]` 阈值要先 `run` 一次刷新它。
  岛群生成器的 `[island]` 也是「default.toml ← 该 run 的快照 ← --set」：**改已有键的默认值，旧 run 仍用快照里的旧值**（先 `run` 刷新，全命中缓存、不到 1 s）；
  **键的含义变了就改键名**（旧键留在快照里无害），否则旧 run 会拿旧含义的数去用。快照写出时中文键要加引号（`dump_toml` 曾把 `汇聚 = 4.0` 写成裸键，读不回来）。
- 提交信息用中文；`out/`、`out-*/` 不提交。

## 架构速查

```
skyisle_gen/
  pipeline.py    阶段注册、缓存 key 链（config[s0k]+shared+skeleton+seed+STAGE_VERSION）、产物 IO
  config.py      TOML 加载/深合并/--set/校验（通过率∈[0,1]、r≤0.98、半衰序 daily≤trade≤migrate≤envoy、eps0>0）
  stages/s01…s10 十步；每步 run(ctx) 读上游产物、写 npz/json + _meta.json
                 ⑨ s09_polity 政治层（第四批 R7：人口、诸邦、采邑、名分/附庸、变法与兼并史），⑩ s10_output 输出（原 ⑨）
  tectonics.py   浮石板块（③ 密度乘子、汇聚核、岛龄；第三批 2）
  localwind.py   ②b 岛对风的扰动（④ 前半：障碍场、带界位移、摩擦/绕流/尾流、可靠局地风；第三批 3）
  moisture.py    上风水汽追踪降水（④ 后半：2° 粗网格显式迎风推进到稳态；第三批 4）
  graph.py       CSR、Dijkstra(heapq)、Brandes 抽样介数、弱连通分量 —— 纯 Python，注意 inf 比较
  weights.py     w_m = λ_ref·cost_m + L_m（L = −ln perm，perm=0 → inf）
  culture.py     World 惰性读取；槽位份额（含本地行）；TV 文化距离；同言线边集
  almanac.py     历法 ↔ 轨道自洽（R1）：纯换算，① 调用，写 planet.json.calendar；check 的 SK-cal 校它
  skeleton.py    骨架第二版的共享定义：D 纬度域、文明核心窗、G 锚定、纬度密度剖面、季节强度公式（全部 |纬度|、随 band_scale 缩放）
  polity.py      政治层产物的只读封装（Polity：邦名/状态/探针行；print_summary），ninegrid/probe/web 共用
  geology.py     地质表现层（R4）：③ 板块网格按节点采样 → 九格表 ① / 探针 / 操作台的叙事文本，不进推导（原则甲）
  check.py       P1–P8 + C1–C5 + IL-*（铁律）+ SK-*（骨架校准，warn-only）
  ninegrid.py    九格表草稿（RegionData 聚合 + build_region_md + lint）
  viz.py / probe.py / web/(server.py bundle.py static/index.html static/vendor/globe.gl.min.js)
                 操作台数据通道：/api/world、/api/fields、/api/grid（② 风 / ④ 气候的 1° 网格场，R2）、/api/texture；单文件版全部内嵌于 INLINE
                 /api/island?run=&node= 按需生成岛群并返回摘要，/api/island/preview 取总览图（探针折叠区「岛群生成器」；单文件版不支持）
                 **岛群调试台** `static/island.html`（`/island.html?run=&node=[&year=]`，探针里有链接）：2D canvas 图层（地形 / 晕渲 / 地表 / 坡度 / 汇流 / 岛号 / 当日海拔温度 / 地形区 / 资源分布（主导，或某一类的赋存品位）；`&base=zone|resource&rk=stone&res=1&fly=行,列,缩放&day=N` 可直接打开；拉远时季相层 / 河道矢量自动降级）、
                 滚轮缩放拖动、悬停读格（高程 / 坡 / 汇流 / 地表 / 水与河宽水深 / 地形区 / 主导资源与各类赋存品位 / 当日温度）、资源点位（采场 / 点 / 片，拉近标赋存区）与漫滩叠加层、主岛河流表（点按钮飞到河口）、约束对照、四季表与图、逐日天气图 + 日期滑杆 / 播放、改年份重生成、`island.*` 参数覆盖重生成；
                 「季相与水情」日图层（积雪 / 雪线、植被枯荣、作物阶段、溪涧按基流水库逐段断流 / 接回、河道涨水漫滩、冰按度日封冻 / 开河、云海漫顶；河道永远画，断流是干河床，四点二十）与「天气特效」（雨雪风暴云雾风粒子）都在浏览器里按逐日天气推，不改产物（DESIGN-NOTES 四点十四）；
                 数据通道 /api/island/data（island.json + climate.json 含 weather.days）、/api/island/raster（terrain.npz 定型数组 base64，> 160 万格抽稀）、POST /api/island/regen
  island/        **第三层岛群生成器**（PLAN-ISLAND，DESIGN-NOTES 四点十四）：`skyisle island <节点>`，按需生成、不进十步管线、不回灌
                 （stages/ 与 check/ninegrid/polity/culture 不得 import 它，pytest 与 IS-iso 有静态断言）
                 __init__  island_config（[island] 段：默认值 ← run 的 resolved ← --set，不进缓存 key）、_node_inputs、build_terrain、generate
                 grid      局部分形噪声（LatticeNoise / FractalNoise，特征尺度以 km 给）、行程并查集连通分量、形态学、块均值 / 双线性、PNG 写出
                 layout    5.1 岛数（n0=30 × 陆地^0.35）、Zipf 大小（总和严格 = area_km2，主岛最大）、角向半径剖面放置（主岛引力、板块走向拉长）、各岛台面高度与目标起伏（岛龄 × 面积^0.3，另一条随机流）、索桥 / 短渡 / 导水槽 MST
                 terrain   5.2 岛形（椭圆 + 域扭曲 + 面积二分反解）、岛龄基形（锥 / 脊 / 台地）+ 幂次定测高曲线、粗网格**隐式河流功率下切**（只切汇流 ≥ 0.3 km² 的河道格、坡面靠休止角；随机流向 + 细网格平滑去方格纹）、
                           仿射拟合（陆地中位 = 台面、峰 − 岸缘 = 目标起伏）；priority_fill / d8 / d8_random / accumulate（5.3 共用）
                 hydro     5.3 河（主岛按 has_river 调阈值）/ 溪涧 / 湖 / 河口盆地、地表 12 类、可耕地按适宜度分位取到 arable_frac；
                           流向在「路由面」上算（填平面 + 弯曲噪声 + 朝岸缘微倾：河在缓坡上蜿蜒、不贴崖边平行跑），湖与抬洼仍按原填平面
                 river     5.3b 河道成形（DESIGN-NOTES 四点十六）：水力几何 w = 5·Q^0.5 × 8、d = 0.35·Q^0.4 × 3（夸张系数设 1 = 真实比例）→ 河宽 ≥ 2 格时加宽；
                           河床下切并向下游单调、河口切豁口成瀑布；两岸压成「漫滩 + 谷坡」剖面（峡谷 32° → 宽谷 9°，随 log Q）；溪涧浅切、按比降接到干流上
                 resources 5.3c 地形区（高山 / 山地 / 丘陵 / 台地平原 / 河谷 / 崖缘 / 水域）+ 13 类资源分三形态（四点十七 / **四点二十一**）：
                           **点**（泉眼、温泉、洞穴）与**片**（林木、泥炭芦苇、浮石露头、鸟粪石）记在 deposits，片占的格写 patch_id（面积一律按它数：sync_resources）；
                           **散**（金属矿、石料、黏土、砂砾、砂金、硫磺）= 赋存场 res_field（uint8 [6,H,W] 品位，可叠）→ 赋存区 occurrences（≥ occ_thr 的连通块）→ 采场 workings
                           （矿坑 / 硫磺坑在这里挑；采石场 / 土坑 / 采砂场 / 淘金点在聚落之后按村挑，tiers.village_workings）。**岩类（矿 / 石 / 硫磺）只在岩类可放区**：
                           山地、高山、裸岩 / 高山草甸、丘陵且坡 ≥ 8°，林坡算（采场格改裸岩），平地林、耕地、湿地、漫滩不算；沉积类赋存可压田，坑不上田不上林（用户拍板）。
                           金属矿是顺板块走向的矿化带、板块内部几乎没有；砂金 = 砂砾 × 上游金 / 铜矿化带的平均品位（hydro 存了群栅格的 D8 下游 recv_i/j 与路由面 route_h）；
                           浮石 = 岛体，崖面与峡谷壁按露头率露出、可开采不影响浮空。resource（uint8）只是显示用的主导类（稀的盖常的）；→ resources.json / png、terrain_zone.png、preview_resources.png
                 climate   5.4 四季：带界随太阳摆动（Δφ = k_shift·倾角·A_sea·cos）取样再缩放到年均；温度 = 年均 + season_range/2·cos(相位 − 滞后)；季型分类命名
                 weather   5.5 逐日：马尔可夫晴雨 + 伽马雨量（风暴日计入预算）、风暴事件、AR(1) 风温、云海漫顶、岸缘 ≤ 0.5 °C 记为雪（小雪 / 大雪 / 暴风雪）；multi_year_stats 供 IS-daily
                 output    5.6 island.json / height.png(16 位) / landcover.png / water.png / arable.png / terrain.npz（含 river_width_m / river_depth_m / floodplain / terrain_zone / resource / res_field / patch_id）/ climate.json / weather_y<年>.csv / preview.png（总览）/ preview_main.png（主岛放大）；
                           water.png 值 6 = 漫滩；河道格的 height 是河床，水面 = 河床 + 水深；rivers.json = 河道中心线折线（调试台 / preview 按河宽画平滑矢量，栅格上河只有一两格宽）；调试台拉近后停手会「细化渲染」当前视口（连续量双线性、分类图扰动 + 加权投票，纯显示）
                 settle    聚落（PLAN-SETTLE，DESIGN-NOTES 四点十五 / 四点十八）：人口只读 ⑨ → 户（10% 非农）；可耕地连通块切田块（k-means 按 80 户地量分）；村址评分；桥头 / 导水槽；蓄水池 / 取水点；
                           前哨、三个主家候选、都与城（城居人口 = 本邑 × 城居率 + 邦 × 集聚率，郭沿索桥，仓城 = 主泊场，祭台）→ settlements.json / png
                 tiers     聚落层级（四点十八）：**飞船取代车船、随处可停 → 没有码头**；专业聚落（矿镇 / 浮石采石村 / 窑村 / 烧炭营 / 温泉地，按资源量、封顶非农 40%）、
                           集镇（中心地：6 km 直线跨岛服务半径、镇距 ≥ 10 km、邑治必为镇）、每个聚落旁一块泊场、村周开垦（林地 → 草坡 / 灌丛薪炭林，林场 patch_id 划掉）、
                           村的采场（开垦之后：石 / 土 / 砂就近取，先合用 1.5 km 内已有的，再在 5 / 2 / 2 km 内按品位 × 距离挑；矿镇矿村改读金属矿赋存区）
                 check     第六节 IS-area/surface/arable/river/channel/season/link/det/iso（硬）+ RES-site/occ/work/geo（硬）/ RES-quarry（软）+ IS-daily（软，60 年）+ SET-pop/field/site/land/town/home（硬）/ SET-water（软）；batch 分层抽样批跑
config/default.toml（所有参数；[web] 段只管操作台显示，不进缓存 key）slots.toml（槽位→模式/阻力档）production_templates.toml（④⑤⑥模板）
```

**节点 = 岛群（R10）**：`islands`/`n_islands`/`area_km2` 等字段名沿用，语义都是「群」——一个节点 = 一个岛群 = 一个邑 = 一个水共同体；群内数十小岛属第三层，不进管线。
关键产物：`s03 islands.npz`（含 `area_km2`：群的总陆地 = 集雨面 = 政治体量；`territory_km2` 势力范围、`land_frac` 陆地占比、`arable_frac` 可用地率；`main_area_km2` 主岛陆地、`wall_m` 岛体墙高 = max(0, height − 300)，第三批 1；`age`/`plate` 岛龄与板块）`/plates.npz`（板块网格）`/cand_edges.npz`（无向候选边，kind 0 kNN/1 远程/2 远征/3 回退）→ `s04 wind_local.npz`（**⑤⑥⑦、check、操作台都从这里读风**：扰动后的 u/v、v_local、obstacle、wake、局部带号）`/band_local.npz`（八条局部带界 edges[8, nlon]，⑤ 的 Φ 与 ⑦ 的窗都按它）`/climate_grid.npz`（precip 由水汽模型算出、q、uplift、conv）`/climate_islands.npz`（含 `catch` = 可用地率×陆地×降水，集雨容量；`has_river`/`river_size` 主岛河流，默认只进九格表文本）→ `s05 perm.npz`（perm[E,4]、perm_no_g、f_regional、g_blocked）→ `s06 routes.npz`（有向 cost[2E]、cost_no_g、cost_m[2E,4]、flow、node_flow、betweenness_sources；前 E 条 a→b 后 E 条 b→a）→ `s07 centers.json/prehist.npz/regions.npz` → `s08 fields.npz`（reach/adopt/strength/share[T,N]、conflict_by、C、L）、`iso.npz`（iso[4,N]、local_share[S,N]）、`traits.resolved.json`
→ `s09 polity.npz`（pop、state[N]（邦 id，稀疏/孤悬为 −1）、polity[N]（含船团/部落，处处 ≥0）、kind、control、dist_cap、fief（−1 直辖 / 采邑之主节点）、realm（兼并后的本朝）、circle、capital[S]）`/polities.json`（诸邦属性、suzerain、reformer、history、fronts、openings）`/history.md` → `s10_output/`（world.json、fig、ninegrid、check）。
**地区 ≠ 邦**：⑦b 的地区只是展示分区（九格表的单位）；邦是 ⑨ 的离散政治单位。九格表 ①③④ 写邑级，⑤⑥⑧ 写邦级，⑨ 文化位置仍是连续场（铁律五不动）。

## 改代码时必须遵守

1. **改了阶段代码就把 `pipeline.STAGE_VERSIONS[k]` +1**，否则旧缓存会被当成命中。只改配置不用改版本。
2. **原则乙**：`s07/s08/s09_polity/ninegrid/polity` 不得出现 `["height_m"]`（check 与 pytest 都有静态断言）。高度只进 s04 温度、s05 落差因子、s06 爬升成本。
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
- 历法（R1，`[s01.calendar]`，almanac.py）：**一年 4 季 × 3 月 × 28 太阳日 = 336 日**（骨架第二版，旧 4 × 28 = 112）→ G 型星 0.96 M☉、0.94 AU、潮汐锁定 73 Gyr；
  卫星朔望月 = 一月（28 日）。历法**不反推带界**。旧版的潮汐锁定张力随之消失（`cal_accept_tidal_tension` 改回 false）。
  `mode = orbit_to_calendar` 反向：给恒星质量 + 轨道半径推每季天数。只写 planet.json，不改任何场。
- 尺度口径 `[shared.scale]`（BACKLOG 第一批拍板）：全世界陆地 25,000,000 km² / 可用地率 0.10 / 100 人/km² 可耕地 → 2.5 亿人。
- 陆地（R8）：`area = 势力范围 × f`，势力范围 = 0.866 × (mean_nn × 500 km)²（与分类共用间距），
  `f = min(0.35, f0 · (ρ/ρ_med)^α · lognormal(σ=0.5))`，α=1，f0 由 Σarea = 25M 二分反解（三 seed 均 ≈0.167）。
  Σ势力范围 ≈ 356M km²（表面 70%），故平均 f ≈ 0.070（BACKLOG 里的 0.107 用的是另一种间距口径，见 DESIGN-NOTES 四点六）。
  α=1 的含义：势力范围 ∝ 1/密度，未封顶的群陆地大致相等（一邑 ≈ 3,000 km²），密接群岛贴顶 0.35 后反而更小（≈1,100 km²）；
  四个地形类的 f 恰落在印尼 0.35 / 菲律宾 0.15 / 夏威夷 0.007 / 孤悬 0.002。
- 可用地率（R9）：`arable_frac` 均值 0.10、对数正态 σ 0.35、夹 [0.03, 0.30]，纯标量、不生成岛内地形。
  集雨容量 `catch = 可用地率 × 陆地 × 降水`（s04）。用处：s06 介数源权重、s07 适宜度（× 陆地规模^γ，**γ=0.5**）、s10 九格表 ①⑤⑧。γ=0 即退回旧式。
  γ 不能取 1：归一化 log 陆地与 log 岛密度的 std ≈0.12–0.13 且相关 ≈ −0.3（旧模型 −0.21），等权会抹平适宜度的地理结构（seed 2026 的 P7 会挂）。R8 换模型后 γ=0.5 三 seed 直接通过，没有重校。
- **政治层（第四批 R7，2026-09-11，`[s09.polity]`）**：人口 = P1 × 可耕地 × clip(降水/**0.25**, 0.25, 1)（≈2.3 亿；骨架第二版由 0.5 改，谷物口径）；控制权重 w = 商旅成本 × (索桥内 1 / 群间飞行 **3**) + R0·L_trade，
  控制力 exp(−w/R)，**R0 = 2.2 天**、θ = 0.2、R 随核心实力^0.25 放大（夹 0.5–2.5）；核心实力 S = 控制范围内 Σ 人口 × 控制力，**立都门槛 = S 中位（`capital_min_strength_frac=1.0`）**，
  0.5 时邦数翻倍、中位只 5 邑。三 seed（骨架第二版）：591/590/607 邦，中位 9–10 邑，密接之都的邦是中疏之都的 3.5–4.8 倍（P8 阈 1.5）。
  宗主半径 = R0 × 0.5 且不随实力放大、王畿锁定（否则中心处人口最稠，宗主反成圈内最大邦）。变法 80 年前、用 20 年、动员 ×3、守方 ×2、宗主顾忌 ×3；
  兼并只由变法之国发动（docs/02 §八 ↔ 铁律三 的调和），先易后难，占领消化按 docs/04 §四 的 [3, 8, 15, 25, 60] 年。来由见 DESIGN-NOTES 四点九。
- **seed 2026 是 P7 的哨兵种子**（拒绝点数只有 seed 42/7 的零头；`k_sub=3` 与 ⑦ 的 `secondary_per_circle=3` 都是为它定的：它的 NE 圈分不到全局峰值）。改 ⑦ 适宜度或次级起源相关的东西，先拿它试。
- **seed 7 是 P6 的哨兵种子**：G 邻域只有 8–13 个岛，枢纽 3–4 个，其中两个是穿 D 干线的门户型（反事实里流量反而上升）。改骨架、板块、绕道弧的东西先拿它试。
- 第三批（2026-09-10）的默认值：板块 `[s03.plates]`（汇聚 ×3、离散 ×0.15、叠层核阈 0.9、D/G 邻域不修饰、乘子归一）、
  障碍增益 `obstacle_gain=4`、带界位移夹 ±3.5°、水汽 `evap_temp_coeff=0.03`/`precip_conv_k=0.6`/τ 4 天、`k_sub=3`。来由见 DESIGN-NOTES 四点八。
- 验收阈值中几个是按三 seed 校准过的：P5 强度比 2.0、重心顺风占比 0.75（起源风速门槛 2 m/s）；C5 中心周边岛上全年温差 ≥ 20 °C、冬温 ≤ 5、夏温 ≥ 20；
  「干旱」口径 `arid_precip` 0.2（≈560 mm）；河流降水门槛 0.22；P6 混合度 0.3 + 坍缩占比 0.25（相对分位只报告）；P7 reach≥0.3、伴随器物≥0.4、地区覆盖 0.2；P3 用聚束障碍分比值 ≥1.5（全局秩相关只参考）。

- **岛群生成器（2026-09-17，PLAN-ISLAND，`[island]`）**：栅格 100 m（群外框 > 2048 格自动加倍，38,000 km² 的最大群落到 400 m）；岛数 12–80、Zipf 1.1、最小岛 0.3 km²；
  岸距 1–15 km（beta(1.3, 2.2)）、索桥 ≤ 2 km 且岸缘高差 ≤ 250 m；**高度**：台面 = height_m（陆地中位），起伏按岛龄对数插值（1000 km² 时新岛 3000 → 老岛 450 m）× (面积/1000)^0.3 × 对数正态 0.25，
  中位分位 新 0.22 / 中 0.2 / 老 0.38（台面离岛底太近时先压到 0.12 再压起伏；台面 < 600 m 的岛底 = 0.5 × 台面）；#1165：岸缘 570 → 峰 1,830 m，坡中位 5°、>15° 占 12%；
  下切 15 / 8 轮在 ≤ 320 格的粗网格上（carve_k 0.3、m 0.5、河道阈 0.3 km²、随机流向 p 1.5、细网格平滑 2 遍）；湖 = 填平深 ≥ 3 m 且 ≥ 0.5 km²（`pit_keep_m=2` 让它少见）；
  常年河按流量：年均 ≥ 0.3 m³/s（1,000 mm 约 19 km²、3,000 mm 约 6、300 mm 约 65；不够则 0.2 × 主岛最大汇流；旧固定 25 km²，四点二十），河宽 / 水深夸张 ×8 / ×3、干流下切 25 m、漫滩 = 10 × 河宽、河谷最远 2 km；
  地形区的局地起伏按 4 km 方窗，山地 ≥ 300 m、丘陵 ≥ 100 m，外加相对高度判据（旧的平岛上只按起伏判一格山地都没有；有真实起伏后主要靠起伏判）；土层到 60° 才归零、坡向湿度修正 ±0.2、成林土层门槛 0.2（按平岛定的 40° / ±0.4 / 0.3 在真实坡上是一片灌丛）；盆地 = 汇流 ≥ max(5 km², 2%) 的河口集水区，≥ 15% 岛面积算大盆地；
  相对降水 → mm：150 + 3850 × p^1.3（第八节 a）；带界摆动 k_shift 0.35（±5° 左右）；季型阈值：四季分明 ≥ 20 °C、冷暖两季 ≥ 8 °C、雨旱 2.5 倍、风暴 / 窗口季差 0.25；
  雨日比例 0.12 + 0.30 × (季雨量/1000)^0.7、湿→湿持续 0.45、伽马形状 0.8、风暴日比例 0.35 × 强度^1.2（布尔覆盖率反解事件数）、云海漫顶只在峰高 < 800 m 的群。
  IS-daily 用 60 年样本：30 年时单季标准误约 5%，和 5% 的阈值同量级（DESIGN-NOTES 四点十四）。
  聚落 `[island.settle]`：5 人/户、村 8–80 户（邑治田块可到 300）、村址离田 ≤ 800 m、村间距 1 km（软）、蓄水池只给 ≥ 5% 岛面积的盆地、
  都的城居率 0.3 / 集聚率 0.05（变法之国 0.10）、郭 5 km；非农 10%、镇距 10 km / 服务半径 6 km、泊场 6 格内坡 ≤ 4°、开垦半径 1.5 km × √(户/40)（1 km 时林地只降到 74%）；
  村取石 5 km（石料按露头算之后林茂的低地 3 km 内常常没有能开的石头；飞船运石）、取土 / 砂 2 km，1.5 km 内已有采场就合用。
  资源 `[island.resources]`（四点二十一）：赋存区阈值 矿 / 硫磺 / 砂金 0.15、黏土 / 砂砾 0.3、石料 0.4，隔一格的碎块算一处、≥ 0.05 km²（不合并时 #1165 有 1,940 区，合并后 736）；
  岩类可放区的丘陵坡门槛 8°（「严格按地表、林地一律不放」在林多的群只剩 1–5% 的地、九成以上的村取不到石头，用户选了按地形判）；
  **可放区 ≠ 石料区**：石料按露头算，坡 12° 起算、30° 满，裸岩 / 高山草甸 / 峡谷壁至少 0.6，林坡 × 0.8，> 40° 算崖面（第一版 8° 林坡就算，#1165 石料区 31%、看上去全岛是石头）；砂金增益 8。
  季型判定的分数 = 差异 / 该项门槛（温度按冷暖两季的 8 °C，不是 20），取 ≥ 1 的最大者；三 seed 全量：四季分明 57–60%、冷暖两季 19–23%、风暴季 18%、雨旱季 1.5%、常夏 1%，
  西风带以北 100% 四季分明，信风带一半冷暖两季一半风暴季。

## 未做 / 可改进（按价值排序）

- Monte Carlo 引擎（`s08.engine="mc"` 只留接口）、软先到权重（`first_arrival_weight` 默认关未实现）
- 季节窗口进模型（现只有 `seasonal` 标志与 ④ 的窗口比例；岛群生成器已给出每季窗口，但管线不读它）；政治性障碍只支持经纬矩形覆盖
- 岛群生成器：资源不进村址选择（村定了再去找石 / 土，矿镇落在矿旁）、赋存区的储量只有面积 × 品位没有吨位、没有地下水与岩性栅格（岩性一岛一种）；河宽是夸张后的数，不是水文模型；不做岛内逐日空间分布；老岛台地的宽谷偏少；可耕地偏向沿河带；`island batch` 的三 seed 统计见 DESIGN-NOTES 四点十四
- 九格表 ④ 特有种 / ⑦ 外观 标【待填】；⑨ 文本量词与归因还比较模板化。⑧ 的世仇 = 接壤且势均力敌、界边最多的邻邦（启发式）
- 政治层：兼并史是静态快照 + 逐邦顺序（无年内事件、无分裂/复国）；附庸只一层；邦名是 `邦NNN` 占位；南圈与 NW 圈不发生变法（docs/11 §八）
- 操作台：路径计算需服务端（单文件版不可用）；边层默认只画流量前 N；无撤销/对比两个 run 的差分视图；气象层没有粒子动画（流线虚线已够用）；水汽 q / 抬升 uplift 已在网格通道里
- 操作台改前端时注意：globe.gl 会清空 `#globe` 的内容，遮罩/悬停框必须放在 `#globeWrap`；`pathsData` 的点高度靠 `pathPointAlt(p=>p[2])` 才生效（DESIGN-NOTES 五）
- 操作台左栏按「视角」组织（地理/气候/航运/文化/政治，`index.html` 的 `LENS`）：新控件或新下拉选项要加 `data-lens="…"` 标明属于哪些视角，否则所有视角都显示；
  新图层开关要登记进 `LAYER_IDS` 并写进相应视角的 `on` 预设。「模式」全局一个（`modeSel`），边相关处用 `edgeModeVal()`（「全部」按商旅）。右侧「开发」标签 = 验收 + 参数。
  地址栏 `#lens=climate&run=seed42&node=1165` 是页面状态：`selectIsland` / `setLens` 都会 `writeHash()`，启动时先读再 `setLens`（否则被重写掉），有 node 就选中并飞过去；顶栏「节点#」回车跳转；岛群调试台的返回链接带 node 回来。
- 性能：8000 岛全跑约 2 分钟（s06 介数 30 s、s10 图 50 s 为大头；s09 政治层约 3 s）；20000 岛未系统测试
- `check --seeds` 多种子批跑、`viz diff`、`config diff` 未做
