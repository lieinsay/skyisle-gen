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
# Python ≥ 3.11，依赖仅 numpy + matplotlib（测试另需 pytest）；默认后端是 C++ 核心库，先编它（见下「C++ 核心库」）
pip install numpy matplotlib pytest nanobind
python core/build.py                             # 约 1 分钟；没编时加 --backend python 用冻结的 Python 参考后端

python -m skyisle_gen.cli run --seed 42          # 全十步：①–⑨ 约 3 s（C++），⑩ 出图约 1 分钟
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
- **⑨ 政治层**（BACKLOG R7）：人口 = 可耕地 × 降水折减；邦从「都城控制力 = exp(−后勤成本/半径)」涌现，短渡可达的边才便宜，
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

## 岛群（第三层，按需生成）

```bash
python -m skyisle_gen.cli island 2051 --run out/seed42            # 一个岛群的全部产物 → out/seed42/islands/2051/（5–40 s）
python -m skyisle_gen.cli island check 2051 --run out/seed42      # IS-* / RES-* / SET-* 校验（含重跑比哈希、势力范围 IS-terr）
python -m skyisle_gen.cli island lod --run out/seed42 --lod-res 2000,1000   # 全行星粗版（原生分辨率生成再降采样，多进程；默认顺带出一年逐日天气，--no-weather 关）
python -m skyisle_gen.cli island floats --run out/seed42 --jobs 28          # 浮高的全行星统计（只跑布局 + 地形）→ islands/float_stats.json / .npz，对标定区间
```

邻群的陆地不许叠：每群与邻群按陆地规模分界、各退半道缝，越界了才重摆（DESIGN-NOTES 四点二十二）。

**浮高**（DESIGN-NOTES 四点二十八，`[island.float]`）：同一群里的岛高低错落——主岛不动，其余岛按岛龄整座上下平移 δ
（比主岛年轻的往上、老的往下，再加随机；往上最多约 +1.5 km、往下最多 −500 m，七成在主岛之上；平移后岸缘不低于 20 m，低台面的群往下挪得少）。
平移在地形那一步、水系之前做，所以高岛的气温（直减率）、地表、高山区、村与资源点的海拔都按新高度；群的四季与逐日天气只看主岛，不变。
产物里：`island.json` 每座岛多一个 `float_m`（δ，m；主岛 0），`surface_m` / `rim_m` / `peak_m` / `keel_m` 与 `terrain.npz` / `height.png` 的高程都**已含** δ
（`cliff_m` 与起伏不变）。岛与岛之间没有索桥（P5 用户定：有更方便的船，架超过 20 m 的桥不合理），群内岛之间全是短渡（飞船），导水槽随之没了；
粗版（`islands_lod/<分辨率>/<节点>.npz`）meta 的各岛也记 `float_m` 与 `age`，浮高之前做的粗版算没做、`island lod` 会重跑。
`--set island.float.enabled=false` 关掉（与改前逐位相同，只多 `float_m` = 0）。

**资源的来历：没有火山**（DESIGN-NOTES 四点三十四，Zhouzhu `docs/PLAN-LAND.md` P3）：岩浆在海下，岛是从海底挣脱出来的拱，顶上带着海底的岩层一起升上来。
所以矿是海底带上来的——热泉硫化物（铜、铅锌、金银、铁帽、少量锡）、蛇纹岩（铬、镍）、锰结核；硫磺是海底热泉沉积的自然硫、温泉是刚出海的余热，两样都只在新岛；
熔岩管换成浮石骨架里的空洞（开在浮石露头旁），火山口删掉；石料的岩性按剥蚀深浅分三层（岛的低处是海相石灰岩的沉积盖层，往高处、岛越老露出辉长岩、蛇纹岩），
采石场可以出石灰岩；海底的沉积层还带来贝壳化石（点）、岩盐（散，赋存场第 7 层）与盐泉（点），每个开了盐井的岩盐区配一个盐井村（专业聚落）。
资源共 16 类：点（泉眼 / 温泉 / 洞穴 / 盐泉 / 贝壳化石）、片（林木 / 泥炭芦苇 / 浮石露头 / 鸟粪石）、散（金属矿 / 石料 / 黏土 / 砂砾 / 砂金 / 硫磺 / 岩盐）。

**地貌 P4：新岛成拱、多核嵌合、谷收拢、湿地、河宽跟局地雨**（2026-09-29，Zhouzhu PLAN-LAND P4 的生成器那一半，DESIGN-NOTES 四点三十五）：
- 新岛（岛龄 < 0.3）不再是火山锥，是从海底挣脱出来的拱：中间最厚、四周最薄，表面平整（`[island.terrain]` 的 `arch_*`、`median_frac_arch`）。
- 汇聚带的一部分大岛由两三个核嵌成（用户定：汇聚带大主岛的三成，`multicore_frac` 0.3；折全行星 ≥ 300 km² 的主岛 17%）：主核是山，小的、老的核是缓降的平原、中间一道干流，
  缝上挤出脊；island.json 各岛的 `cores`（载荷中心、根深……，给游戏以后画几个倒锥的岛底）。
- 谷收拢（用户定「中」，`[island.hydro] capture_reach_km` 3 km）：大谷把 3 km 内小沟的上半截抢过来，山在中间的岛汇水 ≥ 5 km² 出岸缘的河少两三成、前几大盆地的占比升几个点；新岛几乎不收，老岛不切成大峡谷。
- 湿地看「坡缓、离崖缘远、汇来的水排不走」再乘雨（雨的三次方），不拿 800 mm 一刀切：北边平而干的 #6615、#6610 从 0 到一百二十多 km²，南边山群只剩两三 km²，#5498 最多（200 km²）。
- 河的流量、地表湿度、湿地按局地雨算：本岛的起伏（高出本岛岸缘每 km 多 0.4，用户 09-29 定：整座岛浮得高不算，只有山逼着气流抬升才多下雨）× 山脉尺度的迎风坡，
  全群的雨总量不变；terrain.npz 加 `rain_mm`。
- 剖面七群 + #2051 的前后对照（三档设定）在 DESIGN-NOTES 四点三十五；两个后端逐位相同。

**地貌与聚落 P5：有的地方就是没人**（DESIGN-NOTES 四点三十六，Zhouzhu `docs/PLAN-LAND.md` P5 的生成器那半，`island/farmland.py` + `core/src/island/farmland.cpp`）：
- **宜垦 / 已垦拆开**：地表分类只标宜垦（地本身能种：坡 < 25°、最暖的月份 ≥ 10 °C、土层、够湿或近水，`terrain.npz` 的 `cultivable`，剖面八群占陆地 81–91%）；
  已垦（此刻有人种，`cultivated`）由聚落层按人口从最好的宜垦地往外填（好地先占），额度照旧是行星层的可耕率。
- **定居有门槛**：宜垦地的连通片要有水（河 / 湖 / 溪涧 / 泉，或年降水 ≥ 400 mm 能蓄雨）、够一个像样的村（≥ 20 户的地）、能落船（附近有 ≤ 4° 的平地），
  头一轮分到的地不够 20 户的片整片让出来，不再按比例连续撒户。剖面八群 < 30 km² 的小岛常住的从 25–100% 降到 0–62%（多数群 0–9%）。
- **大岛保底**（用户 09-29 定）：主岛以外 ≥ 30 km²、有合门槛的宜垦地的岛，先在它最好的那片地上分一个 20 户的村（户从额度里出，从主岛匀过去），其余照好地先占；
  剖面八群 107 座这样的岛全有村。人没少、只是住处变了：八群总户数不变，住在 < 30 km² 小岛上的从 2.0% 降到 1.1%，主岛 73.5% → 72.9%、别的大岛 24.5% → 26.0%。
- **撂荒**：已垦的最外一层（额度之外、贴着田的 8%）与废村旁的田，带「撂荒了几年」（`fallow_years`），地表按年头是草坡 → 灌丛 → 原本的地表；
  **废村**：让出来的片里头一轮分得最多的几片（每群至多 3 个），标「撤空了几年」。
- **没人住 ≠ 没人用**：光秃小岛上的浮石采石村 / 矿村 / 窑村改成工棚（白天来、晚上走，人算在最近的村），烧炭营季节住；
  没人常住的岛里挑少数放牧 / 夏牧、群边高处的烽火台、邑治附近的庙与墓岛（`settlements.json` 的 `uses`）。
- **荒地有主、领照开垦**（L29，用户定）：`land_tenure` 每岛一条——有农户的岛归就近的村，没人住但有人用的归用它的村的大户，其余官荒归邑（废村的地绝户田入官）。
- **去掉索桥**：`links` 全是短渡，`channels`、`bridgeheads` 删掉；「郭」改成离城 5 km 内、同岛或船程 3 km 内的岛上的村。
- 量剖面：`python docs/probes/transect.py out/seed42/islands 6615,6610,6329,6073,6072,5766,5498,2051 --run out/seed42`（直接读生成器产物，出 PLAN-LAND 第二节同口径的两张表，另有田、人住在哪、有人用的岛）。

**地貌与聚落 P6：水利——谷口的渠和塘、湿地排成圩田**（DESIGN-NOTES 四点三十七，Zhouzhu `docs/PLAN-LAND.md` P6 的生成器那半，`island/waterworks.py` + `core/src/island/waterworks.cpp`，配置 `[island.works]`）：
- 水网是人挖的，放在聚落层（L13）。**谷口的渠**：渠首设在常年河（汇水 ≥ 10 km²）与大溪涧（≥ 30 km²，湿季引水、配一口堰塘）上，从高往低挑——水排到崖边就掉下去了，在出山口、平原上游就截住分走；
  渠是从渠首出去最省工的路合成的树（渠水面按 1/2000 往下降、横穿坡地费工 → 顺等高线走），通到灌区里的田、再按 0.8 km 的格点铺进田里，从谷口散开成扇；
  远处零碎的小块田不值得专门挖渠（新接的渠 ≤ 0.5 km + 3 km × 田的面积才修）。每处渠首一座渠首闸。
- **塘**：每个村一口（灌区里的村塘接渠水，山里的山塘在村上坡的沟里筑坝蓄雨水，平地的村塘接雨水），每圩一口圩塘，季节性的渠首一口堰塘。
- **圩田**：湿地周围 2 km 内的平地在好地先占的第一遍里已经种了过半（好地占完了），才排干围圩——600 m 一格的纵浦横塘（≈ 540 亩一圩）、圩堤、圩塘、圩闸，出水口一座排水闸、一条排水渠接到河；
  圩田算已垦、**在额度之内**（挤掉最外一层的地：已垦 = 额度、人口 = ⑨ 不变），没到压力的湿地照旧是芦苇荡；圩塘那格年降水 < 800 mm 的是泽田（排干了种旱作）。
- 剖面八群：#5498 圩田 38 km²（湿地的 19%）、#6329（开局）2.8 km²、北边 #6610 / #6615 26 / 9 km²（#6610 全是泽田），山里的 #6072、#5766 几乎没有圩田（只 #6072 有 0.5 km²）、以渠和山塘为主；
  谷口的渠每群 92–489 km、灌田占已垦 7–30%（山群一成上下：河谷切得深，渠水面够不着台地上的田）；#6610 没有常年河，渠全从大溪涧引。宜垦 / 已垦 / 撂荒与人住在哪都不变。
- 产物：`settlements.json` 的 `waterworks`（渠首、渠的折线、塘、闸、圩、圩田片与摘要）、`terrain.npz` 的 `polder_id`、`farmland.png` 码 5 圩田、`settlements.png` 码 15 塘 / 16 闸；
  调试台「水利」叠加层；俯视图 `python docs/probes/works_view.py out/seed42/islands/6329 --auto head|patch`；transect.py 多一张水利表。校验多 SET-works。

**地貌与聚落 P6 修订（P6b）：有水利就有人维护；原始地貌与人工地貌分开记**（DESIGN-NOTES 四点三十九，Zhouzhu `docs/PLAN-LAND.md` L30 / L31，用户 09-30 看了 P6 的产物后提的；
`island/waterworks.py`、`island/settle.py` 的 `polder_villages` + `core/src/island/waterworks.cpp`、`settle.cpp`，配置 `[island.works] manage_walk_km / polder_village_blocks / village_head_sep_km`）：
- 次序是「田（含圩田）→ 村址 → 水利」：**每处渠首、渠、塘、闸、圩都记管它的村（`village`），都在那个村走得到的 2 km 以内**（渠的每一点、圩的每一格都算）。
- **圩田自己成村**：圩田按 3 × 3 圩一组另成田块，整组走得到现有的村就挂在那个村上（`villages[].polder_fields`），走不到就在圩里的高处落一个圩村（`villages[].polder`，户从这组圩田来）。
- **渠首一村一堰**：一处渠首只灌管它的那个村的田，渠只走那个村走得到的地方（截短），灌区按格算（渠水面以下、走得到的格），渠首隔 1.5 km；纵浦横塘按两旁的圩分给各自的村，排水渠走出去就截短。
- **废弃的水利**：废村旁一口废塘，照活村的规矩找得到渠首的另有废渠首、废渠（`abandoned: true`、`abandoned_years`、`ruin`，没有管它的村）；废圩不做（废村在旱地上）。
- **原始地貌与人工改造**：`terrain.npz` 加 `landcover_natural`（没有人以前的地表：圩田那格原是湿地）与 `landuse`（0 没动过 / 1 开垦的田 / 2 梯田 / 3 渠灌田 / 4 圩田 / 5 撂荒 / 6 樵牧 / 7 采场），
  `island.json` 的 `landcover` 加原始 / 现状 / 各类改造的面积；调试台底图多「原始地貌（没人以前）」「人工改造」，悬停读「原始：湿地 → 现状：圩田（人工）」。
- 剖面八群：水利离管它的村全在 2 km 内（改前离最近的村：#5498 45%、#6615 48% 的圩超过 2 km，渠首最远 7.3 km）；圩村 33 个、挂着圩田的村 67 个；废塘 13 口；
  原始湿地 = 现状湿地 + 圩田，没人常住的岛上本来就没有湿地（湿地只长在 ≥ 94 km² 的有人的岛上）；已垦 = 额度、人口 = ⑨ 不变。镇只从 ≥ 8 户的村里挑（小圩村不当镇）；
  #6329（开局）、#6072 的邑治跟着换了（P7 的贪心挑镇对户数敏感）。校验多 SET-nature；俯视图 `works_view.py --auto ruin`；transect.py 多「水利离村多远」「原始地貌与人工地貌」两张表。

**水利分级：岛群层只出邑级的大堰与圩区，村级的归营建器**（DESIGN-NOTES 四点四十，Zhouzhu `docs/PLAN-LAND.md` L32 / L33，用户 09-30 定；
`island/waterworks.py` 的 `big_plan` + `core/src/island/waterworks.cpp`，配置 `[island.works] village_works / big_*`）：
- 用户：岛群层的水利得是**都江堰级别**的才值得生成；不同村子修水利的方式不一样，村级的归营建聚落。**邑级大堰**在好地先占**之前**定：堰设在汇水 ≥ 30 km² 的常年河上、抬水 2 m，
  干渠按 0.3 m/km 顺等高线走出至多 25 km（工的算法同村的渠），灌区 = 渠水面以下、坡 < 5° 的宜垦地，与「水够灌的」（汇水 × 年雨 × 径流 / 600 mm）取小；
  灌得到 ≥ 20 km² 且 ≥ 这群额度的 10% 才修，一群至多两处、一个水系一处。灌区的地适宜度 × 2，好地先占先占它——田、村、镇往灌区聚（额度、人口不变）。
- 渠在村落之后修到灌区里在种的地上（1.5 km 格点铺支渠），每个用水的村一个**分水口**（渠上离村最近的格），**邑管**（`big_works[].maintainer`、渠的 `work`）。
  `village_works = false`（默认）：村的渠首与渠、村塘 / 山塘 / 堰塘、圩塘、废塘都不在岛群层出了（`true` 回到 P6b 的样子），圩区照旧；村里的塘、井、水圳是营建器按风格修的。
- 剖面八群：五群有大堰（都在主岛）、灌到的占已垦 12–69%（#6073 69%、#5498 55%、#6615 46%、#2051 24%、#6329 12%），#6610（没有常年河）、#6072 / #5766（山群、平地不够）没有；
  住在主岛 70.9% → 72.3%，散户少三到四成、村少一到三成（村变大），已垦离村反而近了；开局 #6329 的邑治从岛 2 的 19 户小村回到主岛。顺带：大半在灌区里的连通块不再按邑治 300 户切田块。
- 营建器的 `plan.json` 的 `meta.anchors.water` 记穿过窗口的渠与本村的分水口、plan.png 画出来（营建算法暂不避渠）。调试台画大堰（橙菱形）、大堰的渠（橙）与分水口（青方块），
  泊场改成方块 + H；河道平滑时钉住汇流点（之前支流末端被挪开，看着像没接上）。俯视图 `works_view.py --auto big`；transect.py 多「邑级大堰」表。

**地貌与聚落 P7：镇、邑治、航船、中转站**（DESIGN-NOTES 四点三十八，Zhouzhu `docs/PLAN-LAND.md` P7 的生成器那半——PLAN-LAND 的最后一期，`island/market.py` + `core/src/island/market.cpp`，配置 `[island.market]`）：
- 前提（L24，用户定）：船寻常，但能控制的浮石船造起来有门槛，一般人家没有——**本岛走路赶集，跨岛搭定班的航船**。
- **泊场先于镇**：能停很多船、能堆货的大块缓坡平地（坡 ≤ 4° 的空平地，1 km 见方窗里够 0.5 km²，隔 5 km）是 `harbors`，每处记「能停多少船」（平地一半能用、一条船 2000 m²，满窗 302 条）；
  北边平的群一百四十处上下，山群十几二十处。村照旧各有船台（都记能停几条船），镇与邑治另挨着一处大泊场：镇 = 老村核心 + 朝泊场长出来的一条街（`towns[].street`）。
- **镇**：以周围户数为主——本岛 6 km 内按走路算，别的岛只算航船够得着的（船程 = 直线 × 风的系数，顺风便宜、逆风贵），挨着大泊场的加分；
  村归镇：本岛 10 km 内走路去，走不到的搭航船去船程最近的镇。剖面八群「有别的岛上的人来赶集却没有航船」的镇从每群 2–10 个到 0，镇离大泊场中位从 2.7–13 km 到 1.4–3.7 km。
- **航船**（`boat_lines`）：每个镇的航船村按方位分线（一条至多 6 个村、一趟船程 ≤ 60 km），从最远的村出发挨个接人、开到镇上的泊场，三日一班（沿途 ≥ 400 户每日一班）；
  剖面八群每群 8–21 条、227–748 km，连一成半到三成半的村。
- **邑治**：从镇里挑航船汇得最多、靠大泊场的（不按最大田块、不按好守）：八群里七群的邑治换了，多是挨着大泊场、几十户的小村长出来的镇；开局 #6329 恰好还是村001；#5766 落在西边的小岛上。
- **中转站**（`relays`，L26：生成器的功能，游戏先不做）：⑥ 里本群的邻边按方位合成口子，落在群朝那边最外的小岛上——每个口子是关卡（进出本群验货交税），
  按流量、成本、风暴、枢纽再加过夜 / 候风 / 避风 / 换船；群边高处的烽火台（统一 P5 的）也在这里。住守关的、驿卒、开客栈的、修船的、仓夫，不种地、粮靠外运，
  户从非农户里出（剖面八群每群 0–51 户；枢纽群 #6314 的换船站 35 户）。⑥ 的背水群（#2051）只有烽火台。
- 人口照旧 = ⑨：非农户先给专业聚落、再给中转站，余下是镇的市户。宜垦 / 已垦 / 撂荒、田、村不变；水利只有镇所在的村的村塘水面跟着变。
- 营建器第四步「集镇与城」要读的（大泊场、朝泊场的街）备在 `town/site.py` 的锚点里。俯视图 `python docs/probes/market_view.py out/seed42/islands/6329 [--seat]`；
  transect.py 多一张「镇、邑治、航船、中转站」表；调试台「聚落」叠加层画航船线、街、中转站；校验多 SET-market。

## 聚落营建器（独立工具，参考用）

> **只给用户做参考，暂不接进游戏**：产物给人看、做参考；Zhouzhu 游戏不读它，游戏里的村子仍是游戏自己铺的。要接进游戏等用户开口。


给一块地形、一个规模、一种风格，营建出一个建筑群（设计稿 `docs/PLAN-TOWN.md`，各风格的数与出处 `docs/TOWN-SOURCES.md`）。
独立工具，只读岛群生成器的产物（地形地貌、水系、聚落点位、气候），不回写，管线与岛群生成器都不 import 它。风格由用户指定。
**已做四步**（PLAN-TOWN 第一、二、三、五步）：
- **场地地形**：把 100 m 的岛群地形细化到 1–2 m（河按中心线与河宽重刻、岸线与田的边界去方格），或者用十种合成地形 / 外部高程图。
- **华北集村端到端**（第一个风格）：村心、朝向（朝阳 = 朝赤道，南半球院子朝北）、鱼骨街村 / 团块生长两种形态、四合院 / 三合院 / 一字院的宅院成形（门开巽位、门楼、耳房、台地）、
  出村大路与桥、关帝庙、土地庙与村口大树、坑塘与龙王庙、场院、井、泊场；TP 硬项与形态指标自检。
- **十二个内置风格、十个形态算子**：华北集村（鱼骨街村 / 团块生长）、江南水乡（前街后河 / 前河后街的滨水、街市）、徽州宗族村（宗族团块、祠堂）、川西林盘（散居、竹林环绕）、
  岭南梳式（梳式、祠堂、风水塘）、黄土窑洞（沿沟台的靠崖窑 / 塬面的地坑院）、客家土楼（圆楼 / 方楼 / 围龙屋）、北欧农庄（tun 散居农庄、浴房铁匠房离开主院 / 瑞典列村）、
  中欧村落（林地排村 / 街村 / 绿地村 / 团村）、英格兰集村（toft & croft 与背巷、村外敞田 / 绿地村 / 团村）、地中海山城（墙贴墙的团块、小广场）、
  日本村落（砺波散居村与屋敷林 / 宿場町的町家与本陣 / 環濠集落）。算子只在合这块地的里挑（滨水要河、等高线要坡、地坑院等要平地）；**地形 > 合理 > 风格**（用户定）：风格的地面上限住不下全村就放宽再排（坡上修台地），算子的宅地放不下就换风格里最小的院子，地形逼出来的形态偏离（λ、朝向）不算风格没做到。
- **画廊**：同一块地 × 若干风格（或每个能用的算子各一格）拼一张图，每格与 synth 同参数的方案相同。
- **营建调试台** `/town.html`：浏览器里换地、换风格、换算子、改参数就重新营建，图层开关、悬停读每栋房子、单击钉住；岛群调试台里「营建此聚落」（或双击村子）直接打开。
能营建宅院、小庄、村；集镇与城（第四步）、专业聚落与整群批跑（第六步）在后面（第四步要读的镇的大泊场与朝泊场的街 P7 已备在岛群产物与锚点里）。

不给 `--style` 只出地面；给了就营建。产物：`site.npz`（细化后的地面 + 占用图）、`plan.json`（方案：路、桥、宅院、每栋房、设施、户、指标、校验）、
`plan.png`（整个窗口）、`plan-detail.png`（建成区局部）、`plan.svg`（矢量，悬停看每栋房）、`style.resolved.toml`（这次用的完整风格表）。

```bash
python -m skyisle_gen.cli town site 2051 --run out/seed42 --site 村037 --style 华北集村   # 岛群里一个村 → out/seed42/islands/2051/town/村037-华北集村/
python -m skyisle_gen.cli town site 1592 --run out/seed42 --site 城         # 只出地面（聚落名还可以是 散户NNN / 邑治 / 镇NN / 矿村NN …）
python -m skyisle_gen.cli town synth --terrain 河谷 --style 华北集村 [--operator fishbone|organic] [--scale 村|小庄|宅院] [--households 60] [--seed 1] [--lat -30]
python -m skyisle_gen.cli town synth --heightmap h.png --res 1 [--water w.png] [--height-scale 0.01]
python -m skyisle_gen.cli town gallery --terrain 河谷 --styles all [--operators each] [--households 40]   # 画廊 → out/town/gallery/河谷-村40户-s1/gallery.png + gallery.json
python -m skyisle_gen.cli town style list                     # 内置风格；style show 华北集村 打出解析后的整张表
python -m skyisle_gen.cli serve                               # 营建调试台 http://127.0.0.1:8642/town.html?terrain=河谷&style=江南水乡
                                                              #   岛群里的村：/town.html?run=seed42&node=2051&site=村037，或在 /island.html 里双击村子
```

风格文件在 `config/town/styles/`（`base.toml` 是全部键的默认与注释，风格只写不同的键），通用功能目录在 `config/town/functions.toml`；
`--set style.键=值` 临时改风格（如 `--set style.ground.max_slope_deg=8`）。硬项没过也照写产物、在命令行里列出来。

窗口大小按规模与户数定（村约 0.9 km 见方、都约 3 km；`--half` 可改）；窗口只决定裁多大，同一个地方换窗口、换风格地面逐格不变。
「朝阳」按所在半球：#2051 在南纬 31.7°，朝阳 = 朝北。工具参数在 `config/town/town.toml`（`--set town.键=值`）。
河道成形之前生成的旧岛群产物没有 `rivers.json`，会报错让先重跑 `skyisle island <节点>`。

## C++ 核心库（生成器后端，行星计划 P6）

生成器的算法已整体移植成 `core/` 里的 C++17 核心库（不含 Python、不含 Godot；Zhouzhu 以后以子模块只编它），
Python 前端经 nanobind 扩展 `skyisle_gen._core` 调它，命令与产物格式不变。设计稿 `docs/PLAN-CORE.md`，实测 DESIGN-NOTES 四点二十三 – 四点二十六。
**P6a（已做）**：第三层的地形段——布局（含势力范围）、岛形、地形、水系、河道成形。
**P6b（已做）**：第三层其余——资源（点 / 片 / 散）、聚落与层级、四季、逐日天气、粗版降采样；cpp 后端下整群 `generate` 一次在 C++ 里算完，
前端只拼 island.json、写产物、出图、校验。
**P6c（已做）**：管线的 ①–④（行星与历法、风带、岛群分布与板块、局地风与水汽降水与季节强度）；cpp 后端下这四步由 C++ 算、npz / json 照旧由 Python 写，
第三层的行星层输入也由 C++ 直接给。
**P6d（已做）**：管线的 ⑤–⑨（障碍、航路与抽样介数、文明中心与地区、特征扩散、政治层），第三层的人口与邦都也由 C++ 给；**默认后端切到 C++**。
⑩ 输出（九格表、出图）只在 Python。Python 版的算法冻结成**参考后端**（`--backend python`），只作对照，新改动先在 C++ 里做。

构建（要 CMake ≥ 3.20、C++17 编译器；Windows 用 VS 2022 的 MSVC，脚本自己进 x64 环境；Linux 直接 cmake，有 Ninja 用 Ninja）：

```bash
python -m pip install nanobind       # 3.x，装在要用的那个 Python 里
python core/build.py                 # Release → skyisle_gen/_core.cp312-win_amd64.pyd（Linux 为 .so；已 gitignore）
python core/build.py --test          # 另跑 C++ 自检（ctest）；--debug 调试版；--clean 重来
```

用哪个后端由 `[engine] backend` 定（P6d 起默认 `"cpp"`；`"python"` 是冻结的参考后端）：

```bash
python -m skyisle_gen.cli run --seed 42 --backend python --set run.id=py-seed42   # 参考后端：①–⑨ 用 Python 算（对照用）；两个后端的阶段缓存 key 分开，另放一个目录
python -m skyisle_gen.cli island 2051 --run out/seed42 --backend python     # = --set engine.backend=python；check / batch / lod / stats 同样
python -m skyisle_gen.cli island compare --run out/seed42 --sample 30 --jobs 10   # 两个后端对照（统计 + island check + 整套产物逐字节）→ islands/compare.json
python -m skyisle_gen.cli island compare --run out/seed42 --sample 30 --timing    # 整群 generate（不写产物）的用时 → islands/timing.json（--no-python 跳过慢的 python）
```

没有 `[engine]` 段的旧 run 快照（如 P6 之前的 out/seed42）按 python 算阶段 key：在上面照旧 `check` / `viz` / `island` 没问题，
但用默认的 cpp 再 `run --seed 42` 会把 ①–⑩ 当成缓存未命中整个重算（产物逐位不变，只是 `_meta.json` 的 key 与 engine 换了）。

随机流与 numpy 逐位一致（PCG64 / SeedSequence / ziggurat 正态与指数 / 泊松 …），浮点运算次序也照 numpy 与 CPython 做，所以两个后端的整套产物
通常逐字节相同（cpp 后端的 island.json 在 meta 里多一个 `"engine": "cpp"`；逐位一致只在开发机上验过，换 CPU 架构或 C 库可能差一位）；
整群 generate 快 10–20 倍（4 线程最大的群 2 s 内）；行星层 ①–⑨ 三个 seed 的产物两个后端逐位相同：⑤–⑨ python 46 s → C++ 0.8 s，
整条管线 ①–⑨（写产物、不出图）python 43 s → C++ 2.5 s，`planet_run` ①–⑨ 在内存里 1.2 s（游戏新建世界的路径）。扩展没编时选 cpp 会报错并提示构建命令。

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
                  town/（聚落营建器：town.toml 工具参数；functions.toml 通用功能目录；styles/ 风格：base + 十二个内置风格）
core/             C++ 核心库（CMake；build.py 一键构建；include/skyisle/（island/ 第三层、planet/ 行星层 ①–⑨、town/ 聚落营建器）、src/、python/ 绑定、tests/ 自检、
                  tools/ 探 numpy 的 ziggurat 表、third_party/pocketfft 与 np.fft 同一份的 FFT）
skyisle_gen/
  stages/         s01_planet … s10_output（十步；s09_polity 为第四批 R7 的政治层）
  polity.py       政治层产物的只读封装（邦名 / 状态 / 探针行 / 摘要）
  almanac.py      历法 ↔ 轨道自洽（4 季 × 28 太阳日 → 恒星质量 / 轨道半径 / 卫星；反向亦可），① 调用
  geology.py      地质表现层：③ 板块格局 → 九格表 ① / 探针的叙事文本（原则甲：不进推导）
  graph.py        Dijkstra / 抽样介数 / 连通分量（纯 numpy + heapq；参考后端，C++ 版在 core/.../planet/graph.hpp）
  culture.py      槽位份额 / TV 文化距离 / 同言线
  check.py        八条验收 + 气候 + 铁律自检 + 骨架/历法校准
  ninegrid.py     九格表草稿生成（docs/08）
  engine.py       生成器后端开关（[engine] backend）与行星层的 C++ 桥（各步 C++ 对象的缓存、从 npz 读回）
  town/           聚落营建器前端（独立工具，PLAN-TOWN）：配置、裁窗口、风格加载、营建请求、产物、出图、画廊、命令；算法在 core/.../town/；调试台在 web/town_api.py + static/town.html
  probe.py  viz.py  weights.py  noise.py  sphere.py  rng.py  config.py  pipeline.py
tests/            公式单测 + 确定性/缓存链集成测试 + C++ 后端对照（test_core_engine / test_core_p6b / test_core_p6c / test_core_p6d；
                  conftest.py：扩展没编时不依赖 C++ 的测试自动用 python 后端）
```
