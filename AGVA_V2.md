# D-AGVA v2：memory 隔离与地面几何自适应

## 为什么需要这两个开关

AG-ReID.v2 上固定强度的 D-AGVA 有两个潜在问题。第一，C2 地面相机本身带有明显俯拍视角，而固定 D-AGVA 会把所有空中图都向平视比例拉伸，对 C2 可能是反方向。第二，增强特征会写回 cluster memory；Stage 1 的高权重逐样本更新和 Stage 2 的 hard/instance 槽都可能被增强样本主导。

本实现把这两个因素拆成相互独立的 Stage 1 可选实验：增强图仍参与损失和梯度，但可以禁止它更新 Stage 1 memory；增强强度也可以根据随机抽取的地面掩码几何量自适应。全程只使用相机域和 SCHP 掩码，不使用真实身份标签。

## 参数

- `--dynamic-agva-no-memory-update`：Stage 1 中真正发生形变的空中样本不更新 `memory_ir`；样本仍参与原有损失和反向传播。
- `--dynamic-agva-target fixed`：默认值，保持原来的固定强度与随机数消耗顺序。
- `--dynamic-agva-target ground-mixture`：从地面训练掩码几何分布中抽取目标比例，并按几何缺口缩放 D-AGVA 强度。
- `--dynamic-agva-need-scale 0.20`：几何需求 `ref / own - 1` 达到该值时使用完整强度。
- `--dynamic-agva-ground-mask-dir PATH`：地面 SCHP 掩码目录。AG-ReID.v2 未显式指定时复用 `--dynamic-agva-mask-dir`；AG-ReID.v1 和 LAGPeR 必须显式指定。

几何量按掩码前景框的高宽比计算：掩码先以最近邻缩放到图像尺寸，保留前景数不少于 `max(2, 0.02×宽)` 的行和不少于 `max(2, 0.02×高)` 的列，再取这些行列围成的框。地面有效掩码覆盖率低于 90% 会立即报错。

新开关只作用于 `dynamic` 实验的 Stage 1。Stage 2 命令与 `d5a4f86` 的原版 `dynamic` 完全相同，不启用 D-AGVA，也不接收这些新参数；Stage 2 日志名仍保留 `mm`/`gm` 标签，用于区分它读取的 Stage 1 模型。代码中保留了 `ClusterContrastTrainer_PCLMP_NoEMALoss_AGVAMask`，但 runner 默认流程不会使用它。

`bridge`、`confident` 和 `three_domain` 路径会在启动前明确拒绝新开关，避免参数被其他 trainer 静默忽略。

## 日志

启动 ground-mixture 时会显示地面掩码覆盖率和 2,000 次离线抽样摘要，例如：

```text
[D-AGVA-GM] ground mask geometry coverage: 30313/30313 (100.00%)
[D-AGVA-GM] 2000-draw preview: scale=0 ..., mean ..., q25/q50/q75 .../.../..., ground median ...
```

每轮结束显示真实训练批次统计：

```text
[AGVA-MM] epoch E: aerial samples N, augmented A (A/N%), skipped updates S, classes kept unchanged K
```

ground-mixture 即使未打开 memory 隔离也会输出这一行，此时 `skipped updates` 和 `classes kept unchanged` 为 0。日志名在 `d1` 后追加 `mm` 和/或 `gm0.2`；所有新开关保持默认时，原有命令、日志名和训练路径不变。

## AG-ReID.v2 Stage 1 实验

以下正式实验不带 `--allow-log-overwrite`。两张 GPU 并行时应给两个进程传同一个锁文件。

### E1：增强样本不更新 memory

```bash
python train_agreid.py \
  --dataset-key agreidv2 \
  --experiment dynamic \
  --run-mode stage1 \
  --data-dir /data/home/lab338/wjj/dataset/AG-ReID.v2 \
  --logs-dir /data/home/lab338/wjj/logs/agreidv2_main \
  --eps 0.6 \
  --dbscan-backend fast \
  --seed 1 \
  --cluster-lock-file /data/home/lab338/wjj/logs/pclhd_cluster.lock \
  --dynamic-agva-no-memory-update
```

### E2：E1 + 地面几何混合目标

```bash
python train_agreid.py \
  --dataset-key agreidv2 \
  --experiment dynamic \
  --run-mode stage1 \
  --data-dir /data/home/lab338/wjj/dataset/AG-ReID.v2 \
  --logs-dir /data/home/lab338/wjj/logs/agreidv2_main \
  --eps 0.6 \
  --dbscan-backend fast \
  --seed 1 \
  --cluster-lock-file /data/home/lab338/wjj/logs/pclhd_cluster.lock \
  --dynamic-agva-no-memory-update \
  --dynamic-agva-target ground-mixture
```

### E2 冒烟测试

```bash
python train_agreid.py \
  --dataset-key agreidv2 \
  --experiment dynamic \
  --run-mode stage1 \
  --data-dir /data/home/lab338/wjj/dataset/AG-ReID.v2 \
  --logs-dir /data/home/lab338/wjj/logs/smoke_agva \
  --eps 0.6 \
  --dbscan-backend fast \
  --seed 1 \
  --cluster-lock-file /data/home/lab338/wjj/logs/pclhd_cluster.lock \
  --dynamic-agva-no-memory-update \
  --dynamic-agva-target ground-mixture \
  --epochs 1 \
  --iters 20 \
  --allow-log-overwrite
```
