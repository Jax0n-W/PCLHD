# Cross-resolution augmentation verification

Date: 2026-11-15

## Unit tests

Command:

```text
python -m unittest discover -s tests -v
```

Complete result:

```text
test_expected_official_test_count_constant (test_agreid_identity.AGReIDIdentityTests.test_expected_official_test_count_constant) ... ok
test_identity_count_guard (test_agreid_identity.AGReIDIdentityTests.test_identity_count_guard) ... ok
test_invalid_name_fails_instead_of_falling_back_to_p (test_agreid_identity.AGReIDIdentityTests.test_invalid_name_fails_instead_of_falling_back_to_p) ... ok
test_official_and_rewritten_names_match (test_agreid_identity.AGReIDIdentityTests.test_official_and_rewritten_names_match) ... ok
test_official_filename_uses_p_t_a (test_agreid_identity.AGReIDIdentityTests.test_official_filename_uses_p_t_a) ... ok
test_protocol_generator_uses_official_parser (test_agreid_identity.AGReIDIdentityTests.test_protocol_generator_uses_official_parser) ... ok
test_released_score_paths_recover_199_training_ids (test_agreid_identity.AGReIDIdentityTests.test_released_score_paths_recover_199_training_ids) ... ok
test_rewritten_filename_recovers_a_from_encoded_camera (test_agreid_identity.AGReIDIdentityTests.test_rewritten_filename_recovers_a_from_encoded_camera) ... ok
test_same_p_can_represent_different_identities (test_agreid_identity.AGReIDIdentityTests.test_same_p_can_represent_different_identities) ... ok
test_v2_filename_has_raw_and_encoded_camera (test_agreid_identity.AGReIDIdentityTests.test_v2_filename_has_raw_and_encoded_camera) ... ok
test_view_manifest_uses_official_identity (test_agreid_identity.AGReIDIdentityTests.test_view_manifest_uses_official_identity) ... ok
test_manifest_split_and_path_resolution (test_agreidv2_support.AGReIDV2SupportTests.test_manifest_split_and_path_resolution) ... ok
test_modal_mapping (test_agreidv2_support.AGReIDV2SupportTests.test_modal_mapping) ... ok
test_v2_baseline_commands_only_change_dataset_and_logs (test_agreidv2_support.AGReIDV2SupportTests.test_v2_baseline_commands_only_change_dataset_and_logs) ... ok
test_v2_rejects_nonbaseline_experiments (test_agreidv2_support.AGReIDV2SupportTests.test_v2_rejects_nonbaseline_experiments) ... ok
test_eps_all_defaults_to_domain_eps (test_clustering_eps.ClusteringEpsTests.test_eps_all_defaults_to_domain_eps) ... ok
test_internal_engine_defaults_are_point_three (test_clustering_eps.ClusteringEpsTests.test_internal_engine_defaults_are_point_three) ... ok
test_nondefault_eps_isolated_in_log_names (test_clustering_eps.ClusteringEpsTests.test_nondefault_eps_isolated_in_log_names) ... ok
test_omitted_eps_all_keeps_existing_names_and_commands (test_clustering_eps.ClusteringEpsTests.test_omitted_eps_all_keeps_existing_names_and_commands) ... ok
test_runner_forwards_eps_all_only_to_stage2 (test_clustering_eps.ClusteringEpsTests.test_runner_forwards_eps_all_only_to_stage2) ... ok
test_stage1_dbscan_uses_requested_eps (test_clustering_eps.ClusteringEpsTests.test_stage1_dbscan_uses_requested_eps) ... ok
test_stage2_dbscan_uses_requested_eps (test_clustering_eps.ClusteringEpsTests.test_stage2_dbscan_uses_requested_eps) ... ok
test_three_domain_dbscan_uses_requested_eps (test_clustering_eps.ClusteringEpsTests.test_three_domain_dbscan_uses_requested_eps) ... ok
test_200_random_symmetric_matrices_match_exact_labels (test_fast_dbscan.FastDBSCANTests.test_200_random_symmetric_matrices_match_exact_labels) ... ok
test_backend_wrapper_and_runner_routing (test_fast_dbscan.FastDBSCANTests.test_backend_wrapper_and_runner_routing) ... ok
test_constructed_noise_large_cluster_and_boundary_conflicts (test_fast_dbscan.FastDBSCANTests.test_constructed_noise_large_cluster_and_boundary_conflicts) ... ok
test_distance_dump_is_opt_in_and_exact (test_fast_dbscan.FastDBSCANTests.test_distance_dump_is_opt_in_and_exact) ... ok
test_channel_mode_is_fixed_off_in_all_commands (test_release_configuration.ReleaseConfigurationTests.test_channel_mode_is_fixed_off_in_all_commands) ... ok
test_dynamic_agva_is_stage1_only (test_release_configuration.ReleaseConfigurationTests.test_dynamic_agva_is_stage1_only) ... ok
test_final_checkpoint_remains_available (test_release_configuration.ReleaseConfigurationTests.test_final_checkpoint_remains_available) ... ok
test_matrix_is_exactly_two_by_two (test_release_configuration.ReleaseConfigurationTests.test_matrix_is_exactly_two_by_two) ... ok
test_original_stage2_is_cmhard_without_ema_loss (test_release_configuration.ReleaseConfigurationTests.test_original_stage2_is_cmhard_without_ema_loss) ... ok
test_stage1_best_checkpoint_is_default (test_release_configuration.ReleaseConfigurationTests.test_stage1_best_checkpoint_is_default) ... ok
test_three_domain_is_stage2_only (test_release_configuration.ReleaseConfigurationTests.test_three_domain_is_stage2_only) ... ok
test_training_engines_have_no_channel_conversion_calls (test_release_configuration.ReleaseConfigurationTests.test_training_engines_have_no_channel_conversion_calls) ... ok
test_aerial_image_process_runs_before_resolution_process (test_resolution_augment.ResolutionAugmentCoreTests.test_aerial_image_process_runs_before_resolution_process) ... ok
test_cross_domain_builder_never_reads_unsupplied_test_images (test_resolution_augment.ResolutionAugmentCoreTests.test_cross_domain_builder_never_reads_unsupplied_test_images) ... ok
test_downscale_restores_original_size (test_resolution_augment.ResolutionAugmentCoreTests.test_downscale_restores_original_size) ... ok
test_ground_resolution_process_touches_only_second_copy (test_resolution_augment.ResolutionAugmentCoreTests.test_ground_resolution_process_touches_only_second_copy) ... ok
test_height_distribution_uses_only_supplied_train_records (test_resolution_augment.ResolutionAugmentCoreTests.test_height_distribution_uses_only_supplied_train_records) ... ok
test_multi_falls_back_when_patches_do_not_fit (test_resolution_augment.ResolutionAugmentCoreTests.test_multi_falls_back_when_patches_do_not_fit) ... ok
test_multi_rectangles_are_in_bounds_nonoverlapping_and_in_range (test_resolution_augment.ResolutionAugmentCoreTests.test_multi_rectangles_are_in_bounds_nonoverlapping_and_in_range) ... ok
test_same_seed_reproduces_one_worker_loader (test_resolution_augment.ResolutionAugmentCoreTests.test_same_seed_reproduces_one_worker_loader) ... ok
test_same_seed_reproduces_pixels_and_patch_stats (test_resolution_augment.ResolutionAugmentCoreTests.test_same_seed_reproduces_pixels_and_patch_stats) ... ok
test_scale_at_least_one_is_exactly_unchanged (test_resolution_augment.ResolutionAugmentCoreTests.test_scale_at_least_one_is_exactly_unchanged) ... ok
test_default_commands_and_log_names_have_no_resolution_changes (test_resolution_augment.ResolutionAugmentRoutingTests.test_default_commands_and_log_names_have_no_resolution_changes) ... ok
test_enabled_commands_add_only_resolution_options_and_suffix (test_resolution_augment.ResolutionAugmentRoutingTests.test_enabled_commands_add_only_resolution_options_and_suffix) ... ok
test_nondefault_resolution_values_extend_suffix (test_resolution_augment.ResolutionAugmentRoutingTests.test_nondefault_resolution_values_extend_suffix) ... ok
test_bucket_evaluation_does_not_change_overall_metrics (test_resolution_augment.ResolutionBucketTests.test_bucket_evaluation_does_not_change_overall_metrics) ... ok

----------------------------------------------------------------------
Ran 49 tests in 12.880s

OK
Saved ALL clustering distance matrix: C:\Users\Jaxon\AppData\Local\Temp\tmpc_cfxwcs\cluster_dist_all_epoch_1.npy (shape=(6, 6), dtype=float32)
Note: number of gallery samples is quite small, got 6
Note: number of gallery samples is quite small, got 6
Note: number of gallery samples is quite small, got 6
Note: number of gallery samples is quite small, got 6
Note: number of gallery samples is quite small, got 6
```

## Default dry-run byte comparison

The complete stdout was captured before changes and again after changes with
`PYTHONIOENCODING=utf-8`. The before/after files had identical sizes and
SHA-256 hashes:

```text
v1 baseline: 2314 bytes
before = FE5CF4B26D910F86AA432CBA487736E8169552879C5F18468B0C92338E6D8FDD
after  = FE5CF4B26D910F86AA432CBA487736E8169552879C5F18468B0C92338E6D8FDD

v2 baseline: 2367 bytes
before = AFC9A6FA13A6FA6F15D38BACF9A67A2C988C19928665AE021E8BA0B89E182EFD
after  = AFC9A6FA13A6FA6F15D38BACF9A67A2C988C19928665AE021E8BA0B89E182EFD
```

Thus both default dry-run outputs are byte-for-byte unchanged.

## Requested dry runs

### v1 baseline + global, GPU 1 minimal validation

Invocation used `CUDA_VISIBLE_DEVICES=1` and:

```text
python train_agreid.py --dataset-key agreid --experiment baseline --run-mode stage1 --epochs 1 --iters 1 --workers 0 --res-aug-mode global --dry-run
```

Key output:

```text
Run mode: stage1
Resolution augmentation: mode=global p=0.5 domains=auto stages=both min_scale=0.15 patches=3..5 patch_area=0.05..0.2
Resolution log suffix: _resglobal_p0.5
Stage 1 log: agreid_dt_s1_d0_channel_off_seed1_resglobal_p0.5
Stage 2 log: agreid_dt_baseline_d0_t0_channel_off_s2_seed1_resglobal_p0.5

$ python _train_agreid_engine.py --dataset agreid_ir --trial 1 --epochs 1 --iters 1 --step-size 20 --lr 0.00035 --weight-decay 0.0005 --batch-size 64 --height 288 --width 144 --num-instances 16 --eps 0.3 --k1 30 --k2 6 --temp 0.05 --momentum 0.2 --seed 1 --print-freq 50 --pooling-type gem --data-dir C:\home\lab338\Jaxon\dataset\AGreid_train --logs-dir <workspace>\logs_agreid_dt --stage1-log-name agreid_dt_s1_d0_channel_off_seed1_resglobal_p0.5 --stage2-log-name agreid_dt_baseline_d0_t0_channel_off_s2_seed1_resglobal_p0.5 --workers 0 --channel-adap-gray off --debug-nonfinite --stage1-only --res-aug-mode global --res-aug-prob 0.5 --res-aug-domains auto --res-aug-stages both --res-aug-min-scale 0.15 --res-aug-patches 3 5 --res-aug-patch-area 0.05 0.2
```

### v1 baseline + multi

```text
python train_agreid.py --dataset-key agreid --experiment baseline --res-aug-mode multi --dry-run

Resolution log suffix: _resmulti_p0.5
Stage 1 log: agreid_dt_s1_d0_channel_off_seed1_resmulti_p0.5
Stage 2 log: agreid_dt_baseline_d0_t0_channel_off_s2_seed1_resmulti_p0.5
Stage 1 command adds:
  --res-aug-mode multi --res-aug-prob 0.5 --res-aug-domains auto
  --res-aug-stages both --res-aug-min-scale 0.15
  --res-aug-patches 3 5 --res-aug-patch-area 0.05 0.2
Stage 2 checkpoint:
  <workspace>\logs_agreid_dt\agreid_dt_s1_d0_channel_off_seed1_resmulti_p0.5\1\model_best.pth.tar
Stage 2 command adds the same seven resolution options.
```

### v2 baseline + global

```text
python train_agreid.py --dataset-key agreidv2 --experiment baseline --res-aug-mode global --dry-run

Resolution log suffix: _resglobal_p0.5
Stage 1 log: agreidv2_dt_s1_d0_channel_off_seed1_resglobal_p0.5
Stage 2 log: agreidv2_dt_baseline_d0_t0_channel_off_s2_seed1_resglobal_p0.5
Stage 2 checkpoint:
  <workspace>\logs_agreid_dt\agreidv2_dt_s1_d0_channel_off_seed1_resglobal_p0.5\1\model_best.pth.tar
Both generated engine commands add only the seven resolution options and
their values.
```

### Existing v2 baseline checkpoint + resolution buckets

The data/log roots and baseline configuration recorded in `命令行.md` were
used to form this expected checkpoint path and dry run:

```text
python examples/PCLHD_AGReIDv2_test.py \
  --data-dir /data/home/lab338/wjj/dataset/AG-ReID.v2 \
  --resume /data/home/lab338/wjj/logs/agreidv2_main/agreidv2_dt_baseline_d0_t0_channel_off_s2_seed1_eps0.6/1/model_best.pth.tar \
  --res-buckets 80 150 --gpu 1 --dry-run

AG-ReID.v2 evaluation dry run
Protocols: exp1, exp2, exp4, exp5
Resolution buckets: 80 150
Output CSV: <checkpoint-dir>/agreidv2_results_seed1.csv
```

## Preview image status

No AG-ReID training images or baseline checkpoint are present in this Windows
workspace or below `C:\Users\Jaxon\Desktop\实验`. Therefore a real-data PNG
and an actual checkpoint evaluation could not be produced locally. Synthetic
images were deliberately not substituted.

The ready-to-run real-data preview command is:

```text
python tools/visualize_resolution_augment.py \
  --data-dir /path/to/dataset-outer-root \
  --dataset-key agreidv2 \
  --samples-per-domain 8 \
  --output resolution_augment_preview.png
```

The generated sheet has 16 rows (8 aerial, 8 ground) and five columns
(`original`, `global`, `single`, `multi`, `blur`).

## Blur control conversion

The blur control uses Pillow Gaussian blur with `radius = 0.5 / s` after the
same opposite-domain target-height draw and `s < 1` trigger as global mode.
