# EXP-1110 阶段性实验总结（2026-11-10 17:10 CST，更新）

生成物：`exp_1110_sweeps.md` / `exp_1110_sweeps.tex` / `exp_1110_sweeps.csv`（`examples/make_exp_1110_sweep_tables.py` 自动生成，可重复执行）。

指标说明：
- `avg_seg`：每查询平均连续段数（I/O 次数代理），越小越好
- `avg_amp`：页读放大 = 访问页内记录数 / 结果集记录数，越小越好
- `avg_pa`：每查询平均访问页数 ≈ `avg_seg × avg_amp`，综合代价指标
- `query_us`：单查询延迟（μs），由查询阶段总耗时 / 测试查询数换算；**受并发干扰，仅作参考**
- `build_s`：索引构建总耗时（秒），含学习方法的目标评估与参数优化；受并发干扰，相对比较有效
- 表头 `(n=k)` 为该格子纳入的种子数

注：`avg_page_count` 与 `avg_pa` 为同一量（每查询访问页数），不是索引大小。本框架为定长分页，索引页数 = ⌈n/b⌉ 对各方法相同，索引大小差异仅来自曲线元数据（可忽略），故未单独出表——可在论文中一句话说明。

## 1. 数据覆盖（截至 17:10，队列链 16:48 全部完成）

| 实验 | 数据集 | 种子覆盖 | 状态 |
|---|---|---|---|
| 网格分辨率 k | osm_2d, gmm_2d/3d/4d/5d | **n=5（种子 0–4）** | ✅ 完成 |
| 查询窗口面积 | osm_2d, gmm_4d | **n=5** | ✅ 完成 |
| 查询长宽比 | osm_2d, gmm_4d | **n=5** | ✅ 完成 |
| 页大小 B | osm_2d, gmm_4d | **n=5** | ✅ 完成 |
| 训练查询数 | osm_2d | **n=5** | ✅ 完成 |
| 分布偏移 | osm_2d, gmm_4d | **n=5** | ✅ 完成 |
| Meta 方法 | gmm_4d（含大 N 到 20M） | n=5；大 N n=1 | ✅ 完成 |
| 训练采样率 trainper | osm_2d | n=3 | ✅ 完成 |
| Scale-v2（k=8/5/4，120/120 查询） | osm_2d, osm_3d, gmm_4d | 1M/5M: n=3；2M/10M/15M/20M: n=1；gmm_4d meta 部分 n=5 | ✅ 完成 |
| Layer C scale（k=4，20/20 查询） | 同上 + AdaCurve | 1M–20M n=3；30M n=1；gmm_4d 20M 部分方法 N/A | ✅ 完成 |
| 干净计时（无并发重跑） | grid + sel, 种子 0 | 进行中 | 夜间自动运行 |

锚点校验：补充实验与 Layer C 重叠的 162 个结构指标值全部一致，0 处不一致。

## 2. 主要结论（结构指标）

### LACSFC 的优势场景（n=5 最终数据，建议在论文中聚焦这里）
- **osm_3d 段数**：Layer C 全规模下 LACSFC-H/Opt/Meta 的 `avg_seg` 基本为全场最低（例如 15M：Meta 2.417 vs Hilbert 2.900；2M：H 2.467 vs Hilbert 2.717），n=3 结果稳定。
- **osm_2d 大规模段数**：scale-v2 中 LACSFC-Opt 在 5M/10M/15M/20M 均为最低 `avg_seg`（20M：20.03 vs Hilbert 20.75），且 `avg_amp` 与 Hilbert 基本持平（1.299 vs 1.298）——段数优势不牺牲页读放大。
- **osm_2d 段数打平、不丢分**：grid/sel/aspect/page/shift/train 各表中 LACSFC-Opt 的 `avg_seg` 与 Hilbert 基本持平（±3% 内）。
- **构建代价低（对标 LMSFC Table 7）**：scale-v2 osm_2d 上 `build_s`，LACSFC-Opt 仅为 Hilbert 的 ~1.2×（20M：86.3s vs 74.2s），而 LMSFC-BO 为 ~3.9×（286.8s）、PiecewiseSFC 为 ~7.7×（571.3s）。学习类方法中 LACSFC 的构建开销最小，是论文中可以正面强调的点。

### ⚠️ n=5 修正（与 n=3 初判不同）
- **gmm_4d `avg_amp`：LACSFC 并未优于 Hilbert**。k=4 时 Hilbert 5.090±0.181 vs LACSFC-Opt 5.168±0.229（n=5）；此前 n=3 显示的 ~0.2% 优势是种子噪声。论文中不要声称 gmm_4d 上 amp 更优。
- LACSFC-Opt 在所有 2D 表的 `avg_amp` 均高于 Hilbert 5–22%（osm_2d k8：2.109 vs 1.951；r8 长宽比：3.574 vs 2.920），是其真实短板，而非噪声。

### 持平或被反超的场景（写作时避免夸大）
- **2D `avg_amp` 一律 Hilbert 最优**（osm_2d 全部表，n=5 确认；grid k8：1.951 vs Opt 2.109；sel a1e-4：2.203 vs 2.347；shift u-u：1.469 vs 1.590；aspect r8：2.920 vs 3.574；train t240：1.888 vs 2.091）。
- **gmm_4d `avg_amp` 也是 Hilbert 最优**（n=5：k4 5.090 vs Opt 5.168；scale 全规模 Hilbert 最低）。
- **段数–放大率存在明确 trade-off**：osm_2d 上 LMSFC-BO/PiecewiseSFC 的 `avg_seg` 远低于 Hilbert/LACSFC（k10：4.6/3.4 vs 69/70），但 `avg_amp` 更高；osm_3d 上 LACSFC 是唯一同时压低 seg 且 amp 不劣化的方法。建议同时报告 `avg_seg` 与 `avg_amp`，或用 `avg_pa` 作综合指标。
- **LACSFC-Meta 相对 Opt 无稳定增益**：多数格子与 Opt 相当，个别略好/略差；Meta 的卖点应表述为"免逐场景重优化"而非性能提升。
- **LMSFC-BO 对训练采样率不敏感**（osm_2d，p=1–1000：`avg_amp` 1.70–1.84，无单调趋势；trainper 表仅 LMSFC 系列有数据）。

### 异常/需排除
- **AdaCurve**：`avg_amp` 比全场高 10–300×（疑似其按学习分数连续扫描整段的特性），已从最优标记排除，LaTeX 中用 † 标注。
- **BitAllocZ-BO**：个别格子 `query_us` 异常高（如 gmm_2d k10 约 40 ms/query），需人工复核再决定是否写入论文。
- **PiecewiseSFC `avg_seg` 不随 k 变化**（osm_2d k8/k10 均 ~3.3），说明其分段独立于网格，Table 6 中需文字说明。

## 3. 写作时必须声明的局限
1. `query_us` 全部来自并发执行环境，相对排名可能受干扰；干净计时结果出来前不要把延迟当主证据。
2. 部分格子种子数不同（表头有 n 标注），跨列比较时注意。
3. scale-v2 的 2M/10M/15M/20M 目前只有种子 0；结论以趋势为主。
4. AdaCurve、GBO（非 AC 序）已做脚注标记；MVP 重实现方法（LMSFC、PiecewiseSFC）需声明。
5. 本框架不支持插入/更新、R*-tree 对比、索引大小统计——论文中需说明或引用其他实验。
6. n=5 单侧 Wilcoxon 最小 p=0.031，敏感性表现在可以做显著性检验了。

## 4. 剩余任务
队列链已于 **16:48:32 全部完成**（236 个 done，0 失败）。当前仅剩：
- 干净计时 grid+sel 种子 0（无并发、单任务顺序跑，预计深夜/次晨完成）

完成后重跑 `python3 examples/make_exp_1110_sweep_tables.py` 即可刷新 query_us 列为干净值。

日志：`results/exp_1110_sweeps/queue_chain.log`、`results/exp_1110_scale_v2/driver_lane{1,2,3}.log`、`results/exp_1110_clean_timing/driver.log`。
