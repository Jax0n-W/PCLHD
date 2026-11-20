# Cluster-selected bridge matching

`bridge` is an optional Stage-2 experiment for AG-ReID.v1 and AG-ReID.v2.
It preserves the accepted baseline pipeline—one aerial clustering, one ground
clustering, one ALL clustering, `PCLMP_NoEMALoss`, `CMhard`, the second
ALL-memory optimization, EMA updates, evaluation, and checkpoint selection.
Only the aerial prototype used for aerial/ground bipartite matching changes.

Default `baseline`, `dynamic`, `three_domain`, and `full` commands and log
names are unchanged. The bridge path is selected only by `--experiment bridge`
or `--experiment dynamic_bridge`.

## Algorithm

For every Stage-2 epoch:

1. Cluster all aerial images together, and cluster ground and ALL exactly as
   in baseline. AG-ReID.v2 ALL Jaccard receives explicit `[ground=1,
   aerial=2]` modalities.
2. Assign every aerial image a bridge score.
3. Within each non-noise aerial cluster, sort by score and select
   `ceil(bridge_ratio * cluster_size)` near members, with at least one member.
4. Normalize the selected features, average them, and normalize the mean to
   form the cluster's bridge center.
5. Run the baseline-compatible PCLHD BGM with ground full-cluster centers and
   aerial bridge centers. This is the same padded Hungarian and second-pass
   ground fallback used by the baseline Stage-2 PGM block. The mapping applies
   to every member of the aerial cluster, including its unselected far members.
6. Initialize aerial memory from the original full-cluster center, never the
   bridge center. The baseline alternating cross-view loss remains weighted
   by `0.25`.

An additional full-center BGM is computed for diagnostics only. It does not
participate in training.

## Bridge-score parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--bridge-score` | `ground_knn` | `ground_knn`, `view_csv`, or `random` |
| `--bridge-topk` | `10` | Number of ground neighbours averaged by `ground_knn` |
| `--bridge-ratio` | `0.5` | Near fraction selected independently inside each aerial cluster |
| `--bgm-fallback` | `pclhd` | `pclhd` baseline matching or opt-in `nearest` total mapping |
| `--view-score-file` | dataset default | OOF CSV used only by `view_csv` |
| `--view-score-path-mode` | `basename` | CSV-to-training-file matching mode |
| `--log-gt-match-accuracy` | off | Print majority-identity BGM accuracy; never affects training |

`ground_knn` L2-normalizes the current epoch's clustering features and
computes mean top-k cosine similarity in 1,024-row GPU chunks. It never builds
a full all-image similarity matrix.

`view_csv` reads the `p_ground` column. A missing file, duplicate key, missing
image, invalid value, or incomplete aerial coverage is a hard error; there is
no fallback.

`random` uses a local generator seeded with `seed + epoch`, providing a
reproducible control without changing global RNG state.

The default `pclhd` matching exactly follows baseline: when ground clusters
outnumber aerial clusters, the cost matrix is padded with zero columns and
ground rows assigned to padding receive the baseline second Hungarian pass.
When ground clusters are fewer than aerial clusters, both mappings are empty
and that epoch has no cross-view loss. `nearest` selects the previous total
bidirectional mapping with nearest-prototype fallbacks; its generated Stage-2
log name contains `_b1_nnfb` instead of `_b1`.

Setting `--bridge-ratio 1.0` selects every non-outlier member. With the
normalized clustering features this makes every bridge center equal to its
normalized full-cluster center, so bridge matching degenerates to baseline
matching and can be used as a control experiment.

Each epoch logs aerial cluster count, near/far image counts, near/far mean
scores, Hungarian pairs and fallbacks, and the fraction of aerial clusters
whose bridge-center and full-center matches differ. With
`--log-gt-match-accuracy`, filenames are parsed by
`agreid_identity.parse_agreid_name`; majority-identity accuracy is printed for
both mappings and is never consumed by optimization.

## Difference from three-domain matching

`three_domain` partitions aerial images before clustering and trains a chain
`A_far <-> A_near <-> Ground`. Bridge matching never splits clustering or
training batches. It uses one baseline aerial cluster and only selects a
ground-facing prototype inside that cluster. Consequently all aerial images
retain the baseline cross-view supervision budget and one person is not split
into separate far and near cluster spaces.

## AG-ReID.v2 smoke test

The accepted setup uses `eps=0.6`, fast DBSCAN, and the baseline Stage-1 best
checkpoint:

```bash
DATA=/data/home/lab338/wjj/dataset/AG-ReID.v2
LOGS=/data/home/lab338/wjj/logs/agreidv2_main
S1="$LOGS/agreidv2_dt_s1_d0_channel_off_seed1_eps0.6/1/model_best.pth.tar"

python train_agreid.py \
  --dataset-key agreidv2 \
  --experiment bridge \
  --run-mode stage2 \
  --stage1-checkpoint "$S1" \
  --eps 0.6 --dbscan-backend fast \
  --epochs 2 --iters 20 --seed 1 \
  --bridge-score ground_knn --bridge-topk 10 --bridge-ratio 0.5 \
  --bgm-fallback pclhd \
  --log-gt-match-accuracy \
  --data-dir "$DATA" --logs-dir "$LOGS"
```

The log must show `Bridge selection`, `Bridge-center BGM`,
`Full-center diagnostic BGM`, `Bridge/full different aerial matches`, and two
`[GT Match Accuracy]` lines in every epoch.

For the dynamic variant, replace `bridge` with `dynamic_bridge` and use:

```text
agreidv2_dt_s1_d1_channel_off_seed1_eps0.6/1/model_best.pth.tar
```

## Two GPUs with one clustering lock

The optional lock serializes only the memory-heavy Jaccard+DBSCAN section.
Feature extraction and training remain concurrent. Both jobs must use the
same absolute lock-file path.

Terminal 1:

```bash
CUDA_VISIBLE_DEVICES=0 python train_agreid.py \
  --dataset-key agreidv2 --experiment bridge --run-mode stage2 \
  --stage1-checkpoint "$S1" --eps 0.6 --dbscan-backend fast \
  --cluster-lock-file /data/home/lab338/wjj/locks/pclhd-clustering.lock \
  --data-dir "$DATA" --logs-dir "$LOGS"
```

Terminal 2:

```bash
CUDA_VISIBLE_DEVICES=1 python train_agreid.py \
  --dataset-key agreidv2 --experiment dynamic_bridge --run-mode stage2 \
  --stage1-checkpoint "$S1_DYNAMIC" --eps 0.6 --dbscan-backend fast \
  --cluster-lock-file /data/home/lab338/wjj/locks/pclhd-clustering.lock \
  --data-dir "$DATA" --logs-dir "$LOGS"
```

The same path may be supplied through the environment instead:

```bash
export PCLHD_CLUSTER_LOCK=/data/home/lab338/wjj/locks/pclhd-clustering.lock
```

With neither the CLI option nor the environment variable set, locking is a
no-op and all existing behavior is preserved. Logs print lock wait and hold
times. POSIX `flock` releases the lock when the file descriptor closes,
including abnormal process termination.
