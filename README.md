# 空岛行星生成器（skyisle-gen）

一颗浮空岛行星的地形、气候与文明生成器。
原先是 Zhouzhu 游戏项目的 `generator/`，2026-09-20 独立成库并改名；
它依赖的两份规格快照在 `docs/spec/`。

`docs/spec/12-扩散模型.md` 的实现：按十步管线生成行星风系、岛屿分布、气候、障碍、
航线网络、文明中心、政治层（诸邦与兼并史），并用 **Hägerstrand (1968) Model III 变体**把文化特征扩散成
**连续场**——没有文化标签，没有 flood fill，任何一点的「文化」是百余条特征在该点
的强度叠加。

```
strength(t, j) = reach(t, o→j) × adopt(t, j)
  reach  = max over 航线路径 [ Π 障碍通过率(模式) × exp(−λ_t · 路径成本) ]
  adopt  = 1 − resistance(t) × conflict(t, j)      （同槽位对称不动点）
```

## 安装与运行

```bash
# Python ≥ 3.11，依赖仅 numpy + matplotlib（测试另需 pytest）
pip install numpy matplotlib pytest

cd generator
python -m skyisle_gen.cli run --seed 42          # 全十步，约 2–4 分钟
python -m skyisle_gen.cli check --run out/seed42 # 单独重跑验收
```

同一 seed 必产出同一世界（`tests/test_pipeline.py` 逐字节校验）。
产物在 `out/seed42/`：

| 目录 | 内容 |
|---|---|
| `s01_planet … s08_diffusion/` | 各阶段中间产物（npz + json + `_meta.json`），全部可单独可视化 |
| `s09_polity/polity.npz` · `polities.json` · `history.md` | ⑨ 政治层：人口场、诸邦（邦 = 若干邑）、采邑树、宗主 / 附庸、变法之国与兼并纪年（`skyisle polity` 看摘要） |
| `s10_output/world.json` | 世界汇总（行星、风带、障碍、中心、地区、政体） |
| `s10_output/fig/*.png` | 全套图层（见下） |
| `s10_output/ninegrid/*.md` | 各地区九格表草稿（docs/08 格式）+ 溯源 json；①③④ 邑级，⑤⑥⑧ 邦级 |
| `s10_output/check.md` | 验收报告（docs/12 第八节八条现象 + 气候 + 铁律自检 + 骨架/历法校准） |

## 十步管线

```
① 行星参数 → ② 大气环流 → ③ 岛屿分布 → ④ 气候 → ⑤ 障碍识别
→ ⑥ 航线网络 → ⑦ 文明中心 → ⑧ 特征场扩散 → ⑨ 政治层 → ⑩ 输出
```

每步产物是下一步输入，带缓存 key 链：只改 `[s08]` 参数重跑时 ①–⑦ 直接命中缓存
（`skyisle run --explain` 查看命中情况；`skyisle stage 6` 强制从第 6 步重算）。

要点实现（与规格的对应）：

- **⑤ 障碍是选择性过滤器**（D34）：区域障碍（A 赤道永暴带 / B·C 副热带无风带 /
  D 中央宽空域 / F 中纬风暴带）各有一张「四模式 × 通过率」矩阵（默认值即
  docs/11 §六 那张表），用穿越坐标 Φ 归一化——跨越一条带无论走几跳，通过率
  乘积恰为矩阵值。G 定点永暴为解析圆盘，全模式删边（改道型）。
  边局部因子：宽空域 / 密度骤降 / 高度落差（只筛「谁付得起」）/ 政治关卡（配置）。
- **⑥ 顺风廉价逆风昂贵**：成本 = 距离 × 风向因子^α(模式) × 无风惩罚 × 风暴 × 爬升；
  使节/迁徙对风向不敏感（α=0.2，docs/11 E 行）。干线与枢纽由抽样介数得出，
  不依赖文明中心（顺序不可倒）。
- **⑨ 政治层**（BACKLOG R7）：人口 = 可耕地 × 降水折减；邦从「都城控制力 = exp(−后勤成本/半径)」涌现，能架桥短渡的边才便宜，
  所以密接之都的邦三倍于中疏之都；宗主空心化只剩名分；中心 ② 圈密接边缘的变法之国是唯一能兼并的邦（`docs/02 §八` 与铁律三的调和）。
  政体离散、文化连续，两者不互相回写。
- **节点 = 岛群，不是单岛**（BACKLOG R10）：图里的 8000 个节点各是一个岛群 = 一个「邑」=
  一个水共同体；群内数十小岛彼此 5–15 km、半小时可达，属第三层，按需生成，不进管线。
  `[shared.ships]` 的分类阈值量的是**群与群之间**的间距。产物字段沿用 `islands`/`n_islands`/`area_km2`
  等旧名，语义均为「群」：`area_km2` = 群的总陆地。
- **陆地 = 势力范围 × 陆地占比**（R8）：势力范围 = 0.866 × 群间平均间距²，陆地占比
  f = min(0.35, f0·密度^α·抖动)，f0 由全世界陆地 = `shared.scale.total_land_km2`（25,000,000 km²）
  反解。密接 f≈0.35（印尼）、中疏 ≈0.15（菲律宾）、稀疏 ≈0.007（夏威夷）、孤悬 ≈0.002。
  每群另有可用地率 `arable_frac`（均值 0.10，R9），集雨容量 = 可用地率 × 陆地 × 降水。
  口径自洽：25M × 0.10 × 100 人/km² = 2.5 亿人（公元 1 年全球）。
- **⑦ 中心是涌现的**：适宜度 = 降水 × 稳定 × 岛密度 × 岛群陆地^γ（不含高度——原则乙；
  陆地是集雨面不是海拔，docs/02 §六「一群 = 一水共同体 = 一个基本政治单位」），在
  docs/11 定稿的三个骨架窗内取极大；推不出即报错（原则庚）。附史前扩散
  （顺风单向抱石而渡）与地区划分（仅输出用，非文化边界）。
- **⑧ adopt 用对称不动点**：`S_t = R_t(1 − r_t·max_{u≠t} S_u)` Jacobi 迭代，
  r ≤ 0.98 保证收敛；势均力敌处产生**陡而连续**的过渡（同言线的第二种来源）。
  每槽位含一行「本地自有」（强度随隔离度升高——反射型障碍：孤岛文化），
  归一化后每点每槽位是恒正的比例分布（原则己）。
  高隔离连通分量还会生成「本地起源」特征。

## 可视化（调试全靠看中间层）

```bash
python -m skyisle_gen.cli viz wind|islands|scale|climate|barriers|routes|centers|iso --run out/seed42
python -m skyisle_gen.cli viz scale                     # 岛群陆地 + 集雨容量（log 着色）
python -m skyisle_gen.cli viz perm --mode daily        # 某模式的通过率图
python -m skyisle_gen.cli viz trait calendar@north_east # 单特征 reach/adopt/strength 三联图
python -m skyisle_gen.cli viz slot white_hemp           # 槽位比例分布（每值一张）
python -m skyisle_gen.cli viz isogloss [mode]           # 同言线 + 聚束热图
python -m skyisle_gen.cli viz distance 6468             # 从某点出发的文化距离（按模式分面）
```

每张图正常应长什么样：风带图应是木星式横条 + G 漩涡；航线图的干线应沿温带核心
东西延伸并在 G 处南北分流（红星 = 中转岛）；同言线图各特征的环**不重合**，
聚束（深色）只出现在赤道带、G、无风带边界；distance 图里 envoy 面板应比 daily
面板平得多（文书通、口音不通）。反例即错：干线穿过 G、同言线全部叠在一条线上、
相邻两点距离跳变。

## 3D 操作台（动态可视化）

```bash
python -m skyisle_gen.cli serve            # 打开 http://127.0.0.1:8642/
python -m skyisle_gen.cli viz web --run out/seed42   # 导出单文件 viewer.html（无需服务器，无重跑）
```

**完全离线**：globe.gl（MIT）随包内置于 `skyisle_gen/web/static/vendor/`，页面不加载任何远程资源；
单文件导出把它内嵌进 HTML，拷到没有网络的机器上双击即可。

浏览器里是一颗可旋转缩放的 3D 行星（globe.gl，贴图由风/风暴/降水场渲染），三栏操作：

| 栏 | 能做什么 |
|---|---|
| **图层与着色**（左） | 点大小按岛群陆地（可关）；岛群按地形类 / 陆地 / 集雨容量 / 地区 / 槽位份额 / 特征 reach·adopt·strength / 隔离度 / 适宜度 / 降水 / 史前到达 / 流量着色；边按某模式通过率 / 流量 / 所属障碍着色，可筛干线、跨障碍边、不可通边、G 阻断边；同言线（跨 θ 的边）按模式着色；障碍几何、枢纽、中心间干线开关 |
| **探针**（右） | 点岛：属性、隔离度、出边表（四模式通过率 + 障碍）、各槽位比例分布、「传到了但不要」标记；「以此为家」→ 全球按文化距离着色（可按模式过滤）；再点一岛 → 双地按模式的文化距离 + 「画出最优路径」（逐跳成本/通过率/累计 reach） |
| **九格表**（右） | 任一地区的九格表草稿即时生成并渲染；点岛可直接跳转 |
| **参数**（右） | 障碍通过率矩阵、局部因子、半衰日程、阻力三档、骨架、岛数、风成本……改完「重新生成」：后台跑管线（缓存只重算受影响阶段），进度实时显示，完成后自动切到新 run 并显示验收结果 |
| **验收**（右） | 八条现象 + 气候 + 铁律自检的 PASS/FAIL 与数值 |
| **政体**（右） | ⑨ 政治层：三圈宗主、变法之国、兼并纪年、当前战事、开局候选、最大二十邦，都可点跳到都城 |

每次重跑产出一个新的 run 目录（`out/web-seed42` 等），顶栏可在各 run 之间切换对比。

## 探针

```bash
python -m skyisle_gen.cli probe node 6468        # 属性 + 出边通过率 + 各槽位强度表
python -m skyisle_gen.cli probe edge 100 105     # 因子 × 模式分解
python -m skyisle_gen.cli probe path 100 200 --mode daily   # 逐跳累计 reach
python -m skyisle_gen.cli probe trait calendar@north_east --node 6468  # 谁砍掉了 reach
```

## 调参

所有参数在 `config/default.toml`（可用 `--config my.toml` 叠加、`--set a.b.c=v` 覆盖，
生效值写入 `out/<run>/config.resolved.toml`）：

| 你想调 | 改哪里 |
|---|---|
| 障碍通过率 | `[s05.barriers.*]`（就是 docs/11 §六 那张表）、`[s05.local.*]` |
| 采纳阻力三档 | `[s08.resistance_range]` |
| 距离衰减（每模式半衰日程） | `[s08.half_distance_days]`（校验强制 daily ≤ trade ≤ migrate ≤ envoy） |
| 骨架（D 位置、G 半径、绕道岛弧、中心窗） | `[skeleton]` |
| 岛群数 / 密度 / 分类阈值（群间间距） | `[s03.islands]`、`[shared.ships]` |
| 尺度口径（全世界陆地 / 可用地率 / 人口密度） | `[shared.scale]`（25M km² / 0.10 / 100 人/km²，三者自洽 → 2.5 亿人） |
| 行星大小 / 自转 / 倾角 | `[s01.planet]`（默认 `radius_km = 6371`，即地球）、`shared.day_range_km` |
| 岛群陆地分布 | `[s03.islands]` 的 `land_frac_alpha`（f ∝ 密度^α）、`land_frac_cap`（0.35）、`land_frac_sigma`；可用地率 `arable_frac_sigma/range` |
| 陆地在中心涌现中的权重 | `[s07.centers].area_exponent`（γ，默认 0.5；0 = 完全不看陆地） |
| 特征表 | `config/slots.toml`；或写 `config/traits.toml` 手工指定（优先于模板） |
| 验收阈值 | `[check]` |

调参回路：改参数 → `run`（缓存自动只重算受影响阶段）→ `check` → 看对应图层 →
`probe` 定位到点。`check --calibrate` 输出「走三天 / 走十天」处的文化距离中位数，
对照 docs/04 §三 的口径调 λ。

`check` 的 `SK-perm` 是障碍实测穿越率与 docs/11 占位矩阵的对照（只报警）：
A/B/C/D 应基本吻合（D 只取紧贴 D 两缘、在文明核心纬度的节点对）；F 偏低是正常的——穿进稀疏的西风带北段除了带本身还要付宽空域
的代价（草原不只有风暴，还稀疏）。

## 验收

`check` 实现 docs/12 §八 的八条现象（阈值全部外置）：

1. 相邻两地几乎相同（含硬项 P1b：任何边的文化差异必须由该边的成本与通过率解释——铁律五的代码化）
2. 沿航线差异单调累积
3. 同言线互不重合；聚束处 = 障碍所在
4. 隔 D 而有官方航路的两地：文书通、口音不通
5. 单向传播（顺风起源的特征重心顺风偏移；上风传下风易、反向难）
6. G 邻域中转岛是两文明圈的混合体，且其流量在「无 G 的反事实世界」中坍缩
7. 存在 reach 高而 adopt 低的地点（传到了但不要），并注明被谁挡住
8. 政治层：密接之都的邦明显大于中疏之都的邦、宗主不是圈内最大邦、变法之国在密接边缘且已开始兼并、船团不建国

外加二维气候 C1–C4（雨影、带界起伏、干旱比例、河流比例）、铁律自检（浮石/地质零字段、height 不进社会推导、处处有人、处处有政体、share 恒归一）、
骨架与历法校准 SK-*（只报警）。
退出码：0 全过 / 1 现象未达 / 2 违反铁律。

## 与规格的已声明偏离

1. **reach 的路径取「最大 reach 路径」**（`λ·cost − ln perm` 加权最短路），而非
   「物理最短路再乘通过率」。后者在物理最短路穿过 G 时会得到 reach = 0；
   前者与规格在无障碍时等价，有障碍时自动改道（docs/02 §五：障碍改变的不只是
   强度，还有路径）。`s08.fast = true` 切换为每 (origin, mode) 一棵参考 λ 树的
   近似（快 3–5 倍，λ 抖动幅度内几乎无差）。
2. **adopt 的循环依赖用对称不动点解**（conflict 读作同时平衡），「先到者优势」
   不做默认——到达顺序在顺序翻转处会产生不连续跳变，违反铁律五。
   Monte Carlo 引擎（`s08.engine = "mc"`）为保留接口，未实现（规格 §七：可选）。
3. 九格表为**草稿**：④ 特有物种、⑧ 世仇通婚细节、⑦ 外观等超出地理推导范围的
   格子标【待填】，交给政治层 / docs/03。

## 目录

```
config/           default.toml · slots.toml · production_templates.toml · (traits.toml)
skyisle_gen/
  stages/         s01_planet … s10_output（十步；s09_polity 为第四批 R7 的政治层）
  polity.py       政治层产物的只读封装（邦名 / 状态 / 探针行 / 摘要）
  almanac.py      历法 ↔ 轨道自洽（4 季 × 28 太阳日 → 恒星质量 / 轨道半径 / 卫星；反向亦可），① 调用
  geology.py      地质表现层：③ 板块格局 → 九格表 ① / 探针的叙事文本（原则甲：不进推导）
  graph.py        Dijkstra / 抽样介数 / 连通分量（纯 numpy + heapq）
  culture.py      槽位份额 / TV 文化距离 / 同言线
  check.py        八条验收 + 气候 + 铁律自检 + 骨架/历法校准
  ninegrid.py     九格表草稿生成（docs/08）
  probe.py  viz.py  weights.py  noise.py  sphere.py  rng.py  config.py  pipeline.py
tests/            公式单测 + 确定性/缓存链集成测试
```
