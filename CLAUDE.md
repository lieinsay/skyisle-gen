# generator/ 工作约定（Claude 上下文）

这是 `docs/12-扩散模型.md` 的实现：行星地形与文明生成器。**先读本文件，再读 `docs/DESIGN-NOTES.md`（决策与踩坑全记录）。**
上游规格：`../docs/12-扩散模型.md`（规格书）、`../docs/01-设计铁律.md`（硬约束）、`../docs/02-世界与地理.md` §3–5、
`../docs/11-世界总图.md`（骨架定稿）、`../docs/08-地区设计规程.md`（九格表格式）、`../docs/04-社会与变迁.md` §3（四模式）。

## 环境与命令

- Python 3.12：`%LOCALAPPDATA%\Programs\Python\Python312\python.exe`（不在 PATH；PowerShell 里用 `$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"`）。依赖仅 numpy + matplotlib（+ pytest）。**不引入 scipy/networkx/pandas。**
- 一律在 `generator/` 下执行：
  ```
  $py -m zhouzhu_gen.cli run --seed 42            # 九步全跑（约 2 分钟；只改 [s08] 约 15 s）
  $py -m zhouzhu_gen.cli stage 6 --seed 42        # 从第 6 步强制重算
  $py -m zhouzhu_gen.cli check --run out/seed42   # 七条验收 + 铁律自检（exit 0/1/2）
  $py -m zhouzhu_gen.cli viz all|wind|islands|scale|routes|isogloss|slot <id>|trait <id>|distance <node> --run out/seed42
  $py -m zhouzhu_gen.cli probe node <id> | edge a b | path a b --mode m | trait <id> --node j
  $py -m zhouzhu_gen.cli ninegrid --run out/seed42 [--region K]
  $py -m zhouzhu_gen.cli serve                    # 3D 操作台 http://127.0.0.1:8642/（完全离线）
  $py -m zhouzhu_gen.cli viz web --run out/seed42 # 单文件 viewer.html（内嵌 globe.gl）
  $py -m pytest tests -q                          # 26 个测试，约 4 s
  ```
- 验收基线：**seed 42 / 7 / 2026 三个种子 `check` 必须全过（0 硬项 0 软项）**，改动核心公式或默认参数后都要重跑这三个。
- PowerShell 向 `python -c` 传含引号的代码会被破坏：写成脚本文件再跑。
- 产物目录 `out/` 已 gitignore；`config.resolved.toml` 是 `check/viz/probe` 读取配置的来源——改了 `[check]` 阈值要先 `run` 一次刷新它。
- 提交信息用中文；`out/`、`out-*/` 不提交。

## 架构速查

```
zhouzhu_gen/
  pipeline.py    阶段注册、缓存 key 链（config[s0k]+shared+skeleton+seed+STAGE_VERSION）、产物 IO
  config.py      TOML 加载/深合并/--set/校验（通过率∈[0,1]、r≤0.98、半衰序 daily≤trade≤migrate≤envoy、eps0>0）
  stages/s01…s09 九步；每步 run(ctx) 读上游产物、写 npz/json + _meta.json
  graph.py       CSR、Dijkstra(heapq)、Brandes 抽样介数、弱连通分量 —— 纯 Python，注意 inf 比较
  weights.py     w_m = λ_ref·cost_m + L_m（L = −ln perm，perm=0 → inf）
  culture.py     World 惰性读取；槽位份额（含本地行）；TV 文化距离；同言线边集
  check.py       P1–P7 + IL-*（铁律）+ SK-*（骨架校准，warn-only）
  ninegrid.py    九格表草稿（RegionData 聚合 + build_region_md + lint）
  viz.py / probe.py / web/(server.py bundle.py static/index.html static/vendor/globe.gl.min.js)
                 操作台数据通道：/api/world、/api/fields、/api/grid（② 风 / ④ 气候的 1° 网格场，R2）、/api/texture；单文件版全部内嵌于 INLINE
config/default.toml（所有参数；[web] 段只管操作台显示，不进缓存 key）slots.toml（槽位→模式/阻力档）production_templates.toml（④⑤⑥模板）
```

**节点 = 岛群（R10）**：`islands`/`n_islands`/`area_km2` 等字段名沿用，语义都是「群」——一个节点 = 一个岛群 = 一个邑 = 一个水共同体；群内数十小岛属第三层，不进管线。
关键产物：`s03 islands.npz`（含 `area_km2`：群的总陆地 = 集雨面 = 政治体量；`territory_km2` 势力范围、`land_frac` 陆地占比、`arable_frac` 可用地率；`main_area_km2` 主岛陆地、`wall_m` 岛体墙高 = max(0, height − 300)，第三批 1）`/cand_edges.npz`（无向候选边，kind 0 kNN/1 远程/2 远征/3 回退）→ `s04 climate_islands.npz`（含 `catch` = 可用地率×陆地×降水，集雨容量；`has_river`/`river_size` 主岛河流，默认只进九格表文本）→ `s05 perm.npz`（perm[E,4]、perm_no_g、f_regional、g_blocked）→ `s06 routes.npz`（有向 cost[2E]、cost_no_g、cost_m[2E,4]、flow、node_flow、betweenness_sources；前 E 条 a→b 后 E 条 b→a）→ `s07 centers.json/prehist.npz/regions.npz` → `s08 fields.npz`（reach/adopt/strength/share[T,N]、conflict_by、C、L）、`iso.npz`（iso[4,N]、local_share[S,N]）、`traits.resolved.json`。

## 改代码时必须遵守

1. **改了阶段代码就把 `pipeline.STAGE_VERSIONS[k]` +1**，否则旧缓存会被当成命中。只改配置不用改版本。
2. **原则乙**：`s07/s08/ninegrid` 不得出现 `["height_m"]`（check 与 pytest 都有静态断言）。高度只进 s04 温度、s05 落差因子、s06 爬升成本。
   **陆地不受此限**：`area_km2`/`arable_frac` 是集雨面与人口容量（docs/02 §六），可以进社会推导——高度才是「地理决定贵贱」的禁区。`arable_frac` 刻意不从 `height_m` 推（保持这条卫生习惯）。
3. **铁律五**：文化只以 share/strength 浮点场存在。不得从 argmax 派生地区/标签，不得 flood fill。P1b（每条边的 TV 差 ≤ a + b·(λ_max·cost + max L)）是硬项。
4. **原则己**：每岛必须可达（史前扩散全覆盖）、每槽位 share 和为 1（本地行 ε>0 保证）。任何会造出孤岛的改动（采样、边集、G 阻断）都要查 `IL-ji`。
5. 区域障碍必须走 **Φ 穿越归一化**（`s05.node_phi`），不得逐边乘因子（跨带通过率会随岛数指数衰减）。
6. 随机数只从 `rng.stage_rng / entity_rng` 取；列表排序后使用；不迭代 set。
7. 新增参数：写进 `config/default.toml` 对应阶段段落并给注释；操作台参数面板（`index.html` 的 `PARAM_SPEC` 或矩阵区块）按需加。
8. `slots.toml` / `traits.toml` / `production_templates.toml` 不在 `default.toml` 里，通过 `pipeline.STAGE_EXTRA_SECTIONS` 进 ⑧⑨ 的缓存 key。新增这类独立配置文件要同步登记，否则改了不会失效。

## 当前默认值的由来（调参前先看）

- 带界 8/28/36/62°；G = 剪切纬度(28) − 5 = 23°N，经度 = D 中央 (−10)；D = lon [−30, 10] × lat [6, 36]。
- 分类阈值 = 船只参数：桥 0.15 天 / 小船 1 天 / 大船 3 天（1 天 = 500 km），量的是**群与群之间**的间距（群内永远密接）。西风带密度 0.05、极地 0.012 才出稀疏/孤悬。
- 半衰日程：daily 5–10、trade 15–30、migrate 30–60、envoy 40–80 天。阻力：低 .05–.2 / 中 .3–.6 / 高 .7–.95。
- ε0 0.02 → ε_max 0.3（隔离度尺度 3）；k_sub = 2；每高隔离分量 3 条本地起源特征。
- 行星：半径 6371 km（地球）、自转 24 h、倾角 20°、1 日航程 500 km → 绕行 80 日。半径只经 `days_per_rad` 影响所有边的天数。
- 尺度口径 `[shared.scale]`（BACKLOG 第一批拍板）：全世界陆地 25,000,000 km² / 可用地率 0.10 / 100 人/km² 可耕地 → 2.5 亿人。
- 陆地（R8）：`area = 势力范围 × f`，势力范围 = 0.866 × (mean_nn × 500 km)²（与分类共用间距），
  `f = min(0.35, f0 · (ρ/ρ_med)^α · lognormal(σ=0.5))`，α=1，f0 由 Σarea = 25M 二分反解（三 seed 均 ≈0.167）。
  Σ势力范围 ≈ 356M km²（表面 70%），故平均 f ≈ 0.070（BACKLOG 里的 0.107 用的是另一种间距口径，见 DESIGN-NOTES 四点六）。
  α=1 的含义：势力范围 ∝ 1/密度，未封顶的群陆地大致相等（一邑 ≈ 3,000 km²），密接群岛贴顶 0.35 后反而更小（≈1,100 km²）；
  四个地形类的 f 恰落在印尼 0.35 / 菲律宾 0.15 / 夏威夷 0.007 / 孤悬 0.002。
- 可用地率（R9）：`arable_frac` 均值 0.10、对数正态 σ 0.35、夹 [0.03, 0.30]，纯标量、不生成岛内地形。
  集雨容量 `catch = 可用地率 × 陆地 × 降水`（s04）。用处：s06 介数源权重、s07 适宜度（× 陆地规模^γ，**γ=0.5**）、s09 九格表 ①⑤⑧。γ=0 即退回旧式。
  γ 不能取 1：归一化 log 陆地与 log 岛密度的 std ≈0.12–0.13 且相关 ≈ −0.3（旧模型 −0.21），等权会抹平适宜度的地理结构（seed 2026 的 P7 会挂）。R8 换模型后 γ=0.5 三 seed 直接通过，没有重校。
- **seed 2026 是 P7 的哨兵种子**（拒绝点数只有 seed 42/7 的零头）。改 ⑦ 适宜度或次级起源相关的东西，先拿它试。
- 验收阈值中几个是按三 seed 校准过的：P5 强度比 2.0、重心顺风占比 0.75；P6 混合度 0.3 + 坍缩占比 0.25（相对分位只报告）；P7 reach≥0.3、伴随器物≥0.4、地区覆盖 0.2；P3 用聚束障碍分比值 ≥1.5（全局秩相关只参考）。

## 未做 / 可改进（按价值排序）

- Monte Carlo 引擎（`s08.engine="mc"` 只留接口）、软先到权重（`first_arrival_weight` 默认关未实现）
- 季节窗口进模型（现只有 `seasonal` 标志与 ④ 的窗口比例）；政治性障碍只支持经纬矩形覆盖
- 九格表 ④ 特有种 / ⑧ 世仇通婚 / ⑦ 外观 标【待填】；⑨ 文本量词与归因还比较模板化
- 操作台：路径计算需服务端（单文件版不可用）；边层默认只画流量前 N；无撤销/对比两个 run 的差分视图；气象层没有粒子动画（流线虚线已够用），R6 的水汽/雨影量待定案后走同一网格通道
- 操作台改前端时注意：globe.gl 会清空 `#globe` 的内容，遮罩/悬停框必须放在 `#globeWrap`；`pathsData` 的点高度靠 `pathPointAlt(p=>p[2])` 才生效（DESIGN-NOTES 五）
- 性能：8000 岛全跑约 2 分钟（s06 介数 30 s、s09 图 50 s 为大头）；20000 岛未系统测试
- `check --seeds` 多种子批跑、`viz diff`、`config diff` 未做
