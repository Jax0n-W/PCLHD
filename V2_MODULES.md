# AG-ReID.v2：D-AGVA 与三域链式匹配

本分支在 AG-ReID.v2 上开放完整的 2×2 消融：

| experiment | D-AGVA（Stage 1） | 三域匹配（Stage 2） | Stage 1 来源 |
|---|---|---|---|
| `baseline` | ✗ | ✗ | 自己的 Stage 1（d0） |
| `dynamic` | ✓ | ✗ | 新的 Stage 1（d1） |
| `three_domain` | ✗ | ✓ | **复用 baseline 的 d0 Stage 1 best** |
| `full` | ✓ | ✓ | 复用 dynamic 的 d1 Stage 1 best |

v1 的默认路径、日志名和训练行为保持不变。所有新功能都只在
`--dataset-key agreidv2` 且选择对应 experiment 时生效。

## 改动概要

- `experiment_runner.py`：去掉 v2 只能跑 baseline 的限制；v2 的 SCHP 掩码目录、
  图像根目录和 view-score 文件默认从 `--data-dir` 推导（可显式覆盖）；
  三域 Stage 2 命令传 `--dataset agreidv2_ir --dataset-key agreidv2`。
- `_train_agreid_engine.py`：允许 v2 开启 D-AGVA；Stage 1 开始前打印
  `[D-AGVA] mask coverage: 找到/总数`，`--dynamic-agva-strict-masks` 时缺掩码直接报错。
- `_train_agreid_three_domain_engine.py`：允许 v2；ALL 联合聚类的
  modal-invariant Jaccard 显式传入模态（v2 路径里没有 `aerial_modify` 标记，
  不传会退化为普通平均，baseline engine 早已这样处理）；补上缺失的 `import os`
  （原先输出目录已存在时会 NameError）；打印每个相机的 near 数量。
- `view_domain/`：识别 v2 官方文件名，并映射为与 v1 相同的 `c{A}{C}` 相机编码
  （空中 c00/c10/c20，地面 c02/c03/c12/c13/c22/c23）；`build_view_manifest.py`
  支持空中、地面同在一个 `train_all` 目录，按相机分配域，未知相机报错。
- `tools/preflight_agreidv2.py`：新增 `--mask-dir`（逐张核对掩码）和
  `--view-score-file`（核对每张 C0 训练图恰好出现一次）。
- `scripts/prepare_agreidv2_view_scores.sh`：一条命令生成 v2 远近分数。
- `scripts/run_all_ablations.py`：v2 可批量跑四组。
- 新增测试：`tests/test_view_domain_v2.py`、`tests/test_dynamic_agva_coverage.py`，
  并更新 `tests/test_agreidv2_support.py`。

## 默认路径（v2）

以 `--data-dir /data/AG-ReID.v2` 为例：

| 用途 | 默认值 |
|---|---|
| SCHP 掩码 | `/data/AG-ReID.v2_SCHP/train_all`（镜像 train_all 的相对路径，`.png`） |
| D-AGVA 图像根目录 | `/data/AG-ReID.v2/AG-ReID.v2/train_all` |
| 远近分数 | `<repo>/view_domain/outputs/agreidv2/agreidv2_aerial_view_scores_foreground.csv` |

## 第 1 步：SCHP 掩码

```bash
python generate_schp_masks.py \
  --input-dir  <data-dir>/AG-ReID.v2/train_all \
  --output-dir <data-dir>_SCHP/train_all \
  --checkpoint <exp-schp-201908261155-lip.pth> --device cuda --seed 1 --skip-existing
python tools/preflight_agreidv2.py --data-dir <data-dir> --mask-dir <data-dir>_SCHP/train_all
```

## 第 2 步：远近分数（三域匹配需要；约 1–2 小时）

```bash
bash scripts/prepare_agreidv2_view_scores.sh <data-dir> [<mask-dir>] [GPU]
```

流程：预检 → manifest（按 A 分 3 折）→ 前景二分类器（ResNet-18，3 折 × 30 轮，
`--split-mode camera_identity`）→ OOF 打分（`--grouping per-camera --near-ratio 0.5`）
→ 分析图 → 分数预检。v2 的身份包含 A，所以各折天然身份不重叠，严格划分不会删数据；
manifest 会打印 `identities shared across folds: 0` 作为验证。

ResNet-18 的 ImageNet 预训练权重首次运行会由 torchvision 下载到
`~/.cache/torch/hub/checkpoints/`；服务器无外网时请提前放好该文件。

**跑三域 Stage 2 之前先看 `view_domain/outputs/agreidv2/analysis/`**：分数分布、
各相机阈值、高/低分样例。远、近两组看起来没有区别时，不值得花一天跑 Stage 2。

## 第 3 步：冒烟测试（各约 10–20 分钟）

```bash
SMOKE=<logs-dir>/smoke_v2_modules
# D-AGVA Stage 1
python train_agreid.py --dataset-key agreidv2 --experiment dynamic --run-mode stage1 \
  --eps 0.6 --dbscan-backend fast --epochs 2 --iters 20 \
  --data-dir <data-dir> --logs-dir "$SMOKE"
# 三域 Stage 2（加载已有的 baseline Stage 1 best）
python train_agreid.py --dataset-key agreidv2 --experiment three_domain --run-mode stage2 \
  --eps 0.6 --dbscan-backend fast --epochs 2 --iters 20 \
  --stage1-checkpoint <baseline-s1>/1/model_best.pth.tar \
  --data-dir <data-dir> --logs-dir "$SMOKE"
```

日志里应看到：
- D-AGVA：`[D-AGVA] mask coverage: 21217/21217`；
- 三域：`Matched aerial images: 21217`，三个相机 `c00/c10/c20` 各自约一半为 near，
  `==> Clusters epoch 0: Far=…, Near=…, Ground=…, ALL=…`，以及 `Far<->Near BGM` /
  `Near<->Ground BGM` 统计，最后每轮一行 `* Finished epoch`。

## 第 4 步：正式消融（eps 0.6，fast DBSCAN）

```bash
COMMON="--dataset-key agreidv2 --eps 0.6 --dbscan-backend fast \
  --data-dir <data-dir> --logs-dir <logs-dir> --workers 8 --seed 1"
S1_D0=<logs-dir>/agreidv2_dt_s1_d0_channel_off_seed1_eps0.6/1/model_best.pth.tar
S1_D1=<logs-dir>/agreidv2_dt_s1_d1_channel_off_seed1_eps0.6/1/model_best.pth.tar

# C 组：三域，复用 baseline Stage 1
python train_agreid.py $COMMON --experiment three_domain --run-mode stage2 --stage1-checkpoint "$S1_D0"
# B 组：D-AGVA Stage 1 + 普通 Stage 2
python train_agreid.py $COMMON --experiment dynamic --run-mode full --checkpoint-policy best
# D 组：B 组 Stage 1 出来后
python train_agreid.py $COMMON --experiment full --run-mode stage2 --stage1-checkpoint "$S1_D1"
```

日志目录名：

| 组 | Stage 1 | Stage 2 |
|---|---|---|
| B dynamic | `agreidv2_dt_s1_d1_channel_off_seed1_eps0.6` | `agreidv2_dt_dynamic_d1_t0_channel_off_s2_seed1_eps0.6` |
| C three_domain | （复用 d0） | `agreidv2_dt_three_domain_d0_t1_channel_off_s2_seed1_eps0.6` |
| D full | （复用 d1） | `agreidv2_dt_full_d1_t1_channel_off_s2_seed1_eps0.6` |

每轮评估与 baseline 相同：exp4（C→A）Rank-1 选 best。训练完成后用
`examples/PCLHD_AGReIDv2_test.py` 测四协议。
