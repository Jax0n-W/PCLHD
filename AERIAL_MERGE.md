# Stage 2 空中碎片合并

AG-ReID.v1 和 v2 的空中图在 DBSCAN 后容易出现同一身份分散在多个簇的情况。离线分析中，合并空中碎片簇提升了空中 BCubed F 和空中到地面的匹配准确率。地面图的碎片化较少，同样的合并会降低地面聚类质量，因此此功能只处理空中簇。

每轮先用现有 DBSCAN 得到空中标签。对每个空中簇求成员特征的均值并做 L2 归一化，再计算簇中心的余弦相似度。相似度达到阈值的簇之间连边，用并查集按连通分量合并。即使 A 与 C 未达到阈值，只要 A 与 B、B 与 C 都达到阈值，三者就会归为一簇。合并后按各组最小的原标签重新连续编号，离群点保留为 `-1`。

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--aerial-merge` | `off` | `threshold` 使用固定阈值。 `adaptive` 按簇紧致度计算阈值。 |
| `--aerial-merge-tau` | `0.7` | 固定模式下，中心余弦大于或等于该值即连边。 |
| `--aerial-merge-lambda` | `0.9` | 自适应模式下，簇对的阈值为 `lambda × (s_i + s_j) / 2`。簇紧致度 `s_c` 是归一化成员特征与簇中心的平均余弦相似度。 |
| `--aerial-merge-scope` | `labels` | `labels` 将合并结果用于伪标签和关联。 `association` 将合并结果用于跨视角关联。 |

适用实验为 `bridge`、`dynamic_bridge`、`confident` 和 `dynamic_confident`。Stage 2 日志名在 `_channel_off` 前加入固定模式的 `_am0.7` 或自适应模式的 `_ama0.9`，其中数值来自本次配置。批量实验入口支持相同的合并参数。

每轮报告合并前后簇数、减少的簇数、多簇合并组数和最大组大小。自适应模式还报告紧致度中位数。启用 `--log-gt-match-accuracy` 后，诊断会统计同组合并的旧簇两两配对的多数身份一致率，以及已聚类图片合并前后的 BCubed。没有发生合并时，配对正确率显示为 `nan`。身份从文件名读取，只用于诊断。

## 合并作用范围

默认 `labels` 让空中记忆库、空中伪标签、域内对比学习和空地关联都使用合并后的标签。`association` 保留原始 DBSCAN 碎片标签用于空中记忆库、伪标签和域内对比学习，用合并组中心完成 BGM 和置信度筛选。对比这两种配置，可以单独检验合并对空中表示学习的影响。再与 `--aerial-merge off` 对比，可以检验来自跨视角关联的收益。

空中到地面的映射会把同一组里的碎片都指向该组匹配到的地面簇。地面到空中的映射会在已匹配的合并组内选余弦相似度最大的碎片，平局时选编号最小的碎片。置信度保留比例按源侧计算，空中方向用合并组数，地面方向用地面簇数。匹配 GT 准确率使用翻译回碎片后的映射。

`association` 需要完整簇中心。使用 `bridge` 或 `dynamic_bridge` 时显式传入 `--bridge-ratio 1.0`，两个 `confident` 实验会自动使用该值。该作用范围在日志名的合并后缀加 `as`，例如 `_am0.7as`。关闭合并时忽略 scope。

```bash
python train_agreid.py --dataset-key agreidv2 --run-mode stage2 \
  --stage1-checkpoint "$S1" --data-dir "$DATA" --logs-dir "$LOGS" \
  --eps 0.6 --dbscan-backend fast --experiment confident \
  --aerial-merge threshold --aerial-merge-scope association
```

## 消融命令

先将 `DATA`、`LOGS` 和 `S1` 设为数据目录、日志目录和 Stage 1 checkpoint。

Baseline + 合并（示例使用 AG-ReID.v2）

```bash
python train_agreid.py --dataset-key agreidv2 --run-mode stage2 \
  --stage1-checkpoint "$S1" --data-dir "$DATA" --logs-dir "$LOGS" \
  --eps 0.6 --dbscan-backend fast --experiment bridge --bridge-ratio 1.0 \
  --aerial-merge threshold
```

置信度模块 + 合并（AG-ReID.v1，使用 `nearest` 补齐双向映射）

```bash
python train_agreid.py --dataset-key agreid --run-mode stage2 \
  --stage1-checkpoint "$S1" --data-dir "$DATA" --logs-dir "$LOGS" \
  --eps 0.6 --dbscan-backend fast --experiment confident \
  --aerial-merge threshold --bgm-fallback nearest
```

在 v2 上运行第二组时，使用对应的路径和 checkpoint，将数据集切换为 `agreidv2`，并可沿用默认的 `pclhd` 匹配。
