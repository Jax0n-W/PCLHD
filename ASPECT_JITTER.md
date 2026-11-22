# 训练期高宽比抖动

Aspect Jitter 是仅在训练阶段使用的横向形变增强。评估和测试变换保持不变，身份标签不参与增强决策。新开关默认关闭，因此不指定相关参数时，命令、日志名和训练流程与 `26cbc08` 完全一致。

## 为什么使用整体高宽比

离线分析得到三个直接结论：

1. D-AGVA 只改变人体框内部结构，在当前强度下，增强前后的特征余弦相似度约为 0.997，网络几乎感知不到。
2. 网络能明显感知整体高宽比。空中图横向压窄 10% 后，它与可穿戴相机 C3 的特征相似度提升远大于 D-AGVA。
3. 相机间差距并不统一：v2 的 C3 比空中图高瘦约 14%，C2 与空中图接近；v1 地面相机比空中图高瘦约 17%。

增强只缩放宽度，高度和最终画布尺寸不变。横向放大后居中裁剪，横向缩小后左右补零；之后仍执行原来的 `Pad → RandomCrop → RandomHorizontalFlip → ToTensor → Normalize → RandomErasing`。地面的两个训练视图分别独立抽样。

## 模式和参数

- `--aspect-jitter off|both|view|geometry`：默认 `off`。
  - `both`：空中和地面都在窄化、加宽两个方向上做对数均匀抖动。
  - `view`：空中图只压窄，地面图只拉宽，使两种视角彼此靠近。
  - `geometry`：先从掩码实测每个物理相机与空中组的几何差距，只对差距大于 0.02 的相机做自适应增强。
- `--aspect-jitter-max-ratio`：最大横向变化。`both/view` 默认 0.2；`geometry` 默认 0.3，并作为实测 gap 的截断上限。
- `--aspect-jitter-prob`：每张图应用增强的概率，默认 0.5。
- `--aspect-jitter-stages 1|2|12`：生效阶段，默认仅 Stage 1。Stage 2 日志名始终保留标签，以标明它加载的 Stage 1 模型。
- `--aspect-jitter-ground-mask-dir`：`geometry` 的地面 SCHP 掩码目录。v2 默认等于 `--dynamic-agva-mask-dir`；v1/LAGPeR 必须显式提供。

固定模式的日志标签为 `aj{mode}{max_ratio}p{prob}`。几何模式使用 `ajgeop{prob}`。生效阶段不是默认值 1 时，末尾再增加 `s2` 或 `s12`。

## geometry 模式

启动时，对每张图计算：

```text
输入空间人体高宽比 = mask_bbox(H/W) / 原图(H/W)
gap_c = clip(median_c / median_aerial - 1, 0, max_ratio)
```

空中图作为一个组，地面图按文件名中的物理相机号分组。官方文件名读取 `C` 字段；改写文件名读取编码相机号的个位。任何一组的掩码覆盖率低于 80% 都会立即报错。

启动日志会打印类似下表的统计：

```text
[AspectGeometry] group   images   masks   coverage   median   gap
[AspectGeometry] aerial   21217   21217    100.00%    ...    0.0000
[AspectGeometry]      2   21298   21298    100.00%    ...    0.0000
[AspectGeometry]      3    9015    9015    100.00%    ...    0.1400
```

重点检查每组覆盖率、C2 的 gap 是否接近 0，以及 C3 的 gap 是否符合预期。地面相机有 gap 时横向拉宽；空中图按各地面相机的训练图片数加权抽取目标相机，再按对应 gap 横向压窄。抽到无明显 gap 的相机时不形变。

相机标签仅用于训练期增强，测试流程完全不变。geometry 本身不会标记增强样本，也不会屏蔽 memory 更新。它可以与 D-AGVA 同时使用，顺序为先 D-AGVA、再 geometry。

## 预览图

当前阶段启用后，运行目录会生成 `aspect_jitter_preview.jpg`。固定模式包含 4 张空中图和 4 张地面图；geometry 为每个相机组各取 4 张。每行从左到右展示下限、`r=1`、上限。请检查：

- 窄化时人物变高瘦，左右出现对称黑边；
- 加宽时人物变矮宽，画面从中心裁回原宽度；
- 输出尺寸不变，方向与相机组的 gap 一致。

## AG-ReID.v2 Stage 1 实验

以下公共参数用于 J1、J2 和 G1：

```bash
python experiment_runner.py \
  --dataset-key agreidv2 \
  --experiment baseline \
  --run-mode stage1 \
  --eps 0.6 \
  --dbscan-backend fast \
  --seed 1 \
  --cluster-lock-file /data/home/lab338/wjj/logs/pclhd_cluster.lock \
  --logs-dir /data/home/lab338/wjj/logs/agreidv2_main \
  <实验参数>
```

- J1：`--aspect-jitter both`
- J2：`--aspect-jitter view`
- G1：`--aspect-jitter geometry`

正式实验不要添加 `--allow-log-overwrite`。

J2 冒烟：

```bash
python experiment_runner.py \
  --dataset-key agreidv2 --experiment baseline --run-mode stage1 \
  --eps 0.6 --dbscan-backend fast --seed 1 \
  --cluster-lock-file /data/home/lab338/wjj/logs/pclhd_cluster.lock \
  --logs-dir /data/home/lab338/wjj/logs/smoke_aj \
  --aspect-jitter view --epochs 1 --iters 20 --allow-log-overwrite
```

G1 冒烟：

```bash
python experiment_runner.py \
  --dataset-key agreidv2 --experiment baseline --run-mode stage1 \
  --eps 0.6 --dbscan-backend fast --seed 1 \
  --cluster-lock-file /data/home/lab338/wjj/logs/pclhd_cluster.lock \
  --logs-dir /data/home/lab338/wjj/logs/smoke_aj \
  --aspect-jitter geometry --epochs 1 --iters 20 --allow-log-overwrite
```
