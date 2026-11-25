# -*- coding: utf-8 -*-
"""AG-ReID Stage 2 with cluster-selected aerial bridge prototypes.

This entry keeps baseline Stage-2 clustering, CMhard memories, no-EMA-loss
trainer, ALL-memory optimization, EMA update, evaluation and checkpoints.  It
changes only the aerial prototype used by bipartite graph matching.
"""

from __future__ import absolute_import, print_function

import argparse
import os.path as osp
import random
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
from torch.backends import cudnn

from ChannelAug import ChannelRandomErasing
from clustercontrast.bridge_matching import (
    BGM_FALLBACK_MODES, BRIDGE_SCORE_MODES, CROSS_CONFIDENCE_MODES,
    bridge_centers, bridge_scores, bridge_selection_stats,
    cross_retain_ratio, cross_view_mapping_confidences, gt_match_accuracy,
    matching_difference_stats, pclhd_bgm, retain_mapping_by_confidence,
    select_near_members)
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.three_domain import (
    build_bidirectional_bgm, generate_joint_cluster_features,
    joint_cross_modal_flags)
from clustercontrast.trainers_no_ema_loss import \
    ClusterContrastTrainer_PCLMP_NoEMALoss
from clustercontrast.trainers_confident_cross import \
    ClusterContrastTrainer_PCLMP_ConfidentCross
from clustercontrast.utils.cluster_lock import (
    cluster_lock, resolve_cluster_lock_file)
from clustercontrast.utils.clustering import (
    append_stage2_eps_log_suffix, resolve_eps_all)
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.aspect_jitter import (
    add_aspect_jitter_arguments, aspect_jitter_enabled,
    build_aspect_train_transform, geometry_aspect_jitter_enabled,
    prepare_aspect_jitter, validate_aspect_jitter_args)
from clustercontrast.utils.data.aspect_geometry import (
    get_train_loader_color_geometry, get_train_loader_ir_geometry)
from clustercontrast.utils.data.resolution_augment import (
    add_resolution_augment_arguments, append_resolution_log_suffix,
    build_resolution_augmenters, print_resolution_stats,
    reset_resolution_stats, validate_resolution_augment_args)
from clustercontrast.utils.fast_dbscan import \
    dump_cluster_distance_if_requested
from clustercontrast.utils.logging import Logger
from clustercontrast.viewbridge import (
    VIEW_PATH_MODES, load_view_score_map, match_view_score_map)

from _train_agreid_three_domain_engine import (
    _adaptive_cluster, _create_hard_memory, _evaluate,
    _extract_ordered_features, _load_trusted_checkpoint, _require_nonempty,
    check_stage2_output_directory, get_stage2_output_directory)


try:
    import _train_agreid_engine as baseline
    _BASELINE_IMPORT_ERROR = None
except ModuleNotFoundError as error:  # CPU parser/tests may not have faiss.
    baseline = None
    _BASELINE_IMPORT_ERROR = error


DEFAULT_STAGE2_LOG_NAME = "agreid_bridge_release_s2"
BRIDGE_CHUNK_SIZE = 1024


def _require_baseline():
    if baseline is None:
        raise RuntimeError(
            "formal bridge training requires baseline dependencies; "
            "current missing dependency: {}".format(_BASELINE_IMPORT_ERROR))
    return baseline


def _resolve_project_path(path):
    if not path or osp.isabs(path):
        return path
    return osp.join(osp.dirname(osp.abspath(__file__)), path)


def _validate_args(args):
    if args.channel_adap_gray != "off":
        raise ValueError("Channel augmentation is fixed off in this release")
    if args.bridge_score not in BRIDGE_SCORE_MODES:
        raise ValueError("unknown --bridge-score: {}".format(
            args.bridge_score))
    if args.bridge_topk < 1:
        raise ValueError("--bridge-topk must be at least 1")
    if not 0.0 < args.bridge_ratio <= 1.0:
        raise ValueError("--bridge-ratio must be in (0, 1]")
    if args.bgm_fallback not in BGM_FALLBACK_MODES:
        raise ValueError("unknown --bgm-fallback: {}".format(
            args.bgm_fallback))
    if not 0.0 < args.cross_retain_start <= args.cross_retain_end <= 1.0:
        raise ValueError(
            "cross retention must satisfy 0 < start <= end <= 1")
    if args.cross_retain_ramp_epochs < 1:
        raise ValueError("--cross-retain-ramp-epochs must be at least 1")
    if args.cross_confidence not in CROSS_CONFIDENCE_MODES:
        raise ValueError("unknown --cross-confidence: {}".format(
            args.cross_confidence))
    if args.confident_cross and abs(args.bridge_ratio - 1.0) > 1e-12:
        raise ValueError(
            "confidence-aware cross matching requires --bridge-ratio 1.0")
    if args.bridge_score == "view_csv":
        path = _resolve_project_path(args.view_score_file)
        if not path or not osp.isfile(path):
            raise FileNotFoundError(
                "view_csv bridge score file not found: {}".format(path))
    if (args.dump_cluster_dist_epoch < -1
            or args.dump_cluster_dist_epoch >= args.epochs):
        raise ValueError(
            "--dump-cluster-dist-epoch must be -1 or smaller than --epochs")


def _load_csv_scores(args, aerial_train, dataset_ir):
    if args.bridge_score != "view_csv":
        return None
    score_path = _resolve_project_path(args.view_score_file)
    score_map, _ = load_view_score_map(
        score_path, score_column="p_ground", near_ratio=args.bridge_ratio,
        weight_mode="none", path_mode=args.view_score_path_mode,
        relative_root=args.data_dir)
    matched, stats = match_view_score_map(
        [sample[0] for sample in aerial_train], score_map,
        path_mode=args.view_score_path_mode, relative_root=args.data_dir,
        image_root=dataset_ir.images_dir)
    if stats["missing"]:
        raise ValueError(
            "view_csv must match every aerial image; matched={}, missing={}, "
            "first_missing={}".format(
                stats["matched"], stats["missing"],
                stats["missing_paths"][0]))
    print("Bridge view CSV: {}".format(score_path))
    print("Bridge view CSV matched: {}/{}".format(
        stats["matched"], stats["training_images"]))
    return [float(matched[sample[0]]["raw_score"])
            for sample in aerial_train]


def _build_transforms(args):
    """Build the same Stage-2 transforms as the baseline engine."""
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if aspect_jitter_enabled(args, 2):
        ground = build_aspect_train_transform(
            args, 2, "ground", args.height, args.width, normalizer)
        ground_aug = build_aspect_train_transform(
            args, 2, "ground", args.height, args.width, normalizer)
        aerial = build_aspect_train_transform(
            args, 2, "aerial", args.height, args.width, normalizer,
            crop_size=(288, 144))
        return ground, ground_aug, aerial
    ground = T.Compose([
        T.Resize((args.height, args.width), interpolation=3), T.Pad(10),
        T.RandomCrop((args.height, args.width)),
        T.RandomHorizontalFlip(p=0.5), T.ToTensor(), normalizer,
        ChannelRandomErasing(probability=0.5)])
    ground_aug = T.Compose([
        T.Resize((args.height, args.width), interpolation=3), T.Pad(10),
        T.RandomCrop((args.height, args.width)),
        T.RandomHorizontalFlip(p=0.5), T.ToTensor(), normalizer,
        ChannelRandomErasing(probability=0.5)])
    aerial = baseline.build_ir_train_transform(
        args, args.height, args.width, normalizer)
    return ground, ground_aug, aerial


def _format_accuracy(name, stats):
    accuracy = stats["accuracy"]
    accuracy_text = ("nan" if not np.isfinite(accuracy)
                     else "{:.2%}".format(accuracy))
    print("[GT Match Accuracy] {}: {} ({}/{}, missing={})".format(
        name, accuracy_text, stats["correct_pairs"],
        stats["evaluated_pairs"], stats["missing_pairs"]))


def _compute_epoch_bridge_scores(args, features_aerial, features_ground,
                                 csv_scores, epoch):
    """Run production ground-KNN scoring on GPU; keep other modes on CPU."""
    score_aerial = features_aerial
    score_ground = features_ground
    uses_cuda = args.bridge_score == "ground_knn"
    if uses_cuda:
        score_aerial = features_aerial.cuda(non_blocking=True)
        score_ground = features_ground.cuda(non_blocking=True)
        print("Bridge ground_knn: device={}, chunk_size={}".format(
            score_aerial.device, BRIDGE_CHUNK_SIZE))
    scores = bridge_scores(
        args.bridge_score, aerial_features=score_aerial,
        ground_features=score_ground, topk=args.bridge_topk,
        chunk_size=BRIDGE_CHUNK_SIZE, csv_scores=csv_scores,
        seed=args.seed, epoch=epoch)
    if uses_cuda:
        scores = scores.cpu()
        del score_aerial, score_ground
    return scores


def _match_centers(args, ground_centers, aerial_centers):
    if args.bgm_fallback == "pclhd":
        return pclhd_bgm(ground_centers, aerial_centers)
    return build_bidirectional_bgm(ground_centers, aerial_centers)


def main_worker_stage2(args):
    _require_baseline()
    start_time = time.monotonic()
    data_dir = args.data_dir
    args.memorybank = "CMhard"
    args.logs_dir = get_stage2_output_directory(
        args.stage2_log_name, args.trial, logs_root=args.logs_root)
    cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, "{}log.txt".format(
        args.trial)))
    print("==========\nArgs:{}\n==========".format(args))
    print("Stage 2 bridge matching enabled: True")
    print("Stage 2 memory bank: CMhard")
    print("Stage 2 EMA loss included: False")
    print("Stage 2 EMA model update enabled: True")
    print("Stage 2 ALL-memory optimization enabled: True")

    ir_name, rgb_name = baseline.get_training_dataset_names(args.dataset_key)
    dataset_ir = baseline.get_data(ir_name, args.data_dir, trial=args.trial)
    dataset_ground = baseline.get_data(
        rgb_name, args.data_dir, trial=args.trial)
    aerial_train = sorted(dataset_ir.train)
    ground_train = sorted(dataset_ground.train)
    all_train = ground_train + aerial_train
    csv_scores = _load_csv_scores(
        args, aerial_train, dataset_ir)
    resolution_augmenters = build_resolution_augmenters(
        args, aerial_train, ground_train, stage=2,
        aerial_root=dataset_ir.images_dir,
        ground_root=dataset_ground.images_dir)
    prepare_aspect_jitter(
        args, 2, aerial_train, ground_train,
        dataset_ir.images_dir, dataset_ground.images_dir, args.logs_dir)

    model, model_ema = baseline.create_model(args)
    checkpoint = _load_trusted_checkpoint(args.stage1_checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model_ema.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.Adam(
        [{"params": [value]} for _, value in model.named_parameters()
         if value.requires_grad], lr=args.lr,
        weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=0.1)
    trainer_class = (ClusterContrastTrainer_PCLMP_ConfidentCross
                     if args.confident_cross
                     else ClusterContrastTrainer_PCLMP_NoEMALoss)
    trainer = trainer_class(model, model_ema)

    clusterers = baseline.build_agreid_stage2_clusterers(args)
    cluster_aerial = clusterers["ir"]
    cluster_ground = clusterers["rgb"]
    cluster_all = clusterers["all"]
    print("Aerial/Ground clustering initial eps: {:.3f}".format(args.eps))
    print("ALL clustering initial eps: {:.3f}".format(cluster_all.eps))

    best_r1 = 0.0
    best_map = 0.0
    best_epoch = -1
    iters = args.iters if args.iters > 0 else None

    for epoch in range(args.epochs):
        reset_resolution_stats(resolution_augmenters)
        with torch.no_grad():
            print("==> Extract Ground features")
            features_ground_ema = _extract_ordered_features(
                model_ema, dataset_ground, ground_train, args, mode=1)
            features_ground = _extract_ordered_features(
                model, dataset_ground, ground_train, args, mode=1)
            print("==> Extract Aerial features")
            features_aerial_ema = _extract_ordered_features(
                model_ema, dataset_ir, aerial_train, args, mode=2)
            features_aerial = _extract_ordered_features(
                model, dataset_ir, aerial_train, args, mode=2)
            features_all = torch.cat(
                (features_ground, features_aerial), dim=0)

            with cluster_lock(
                    args.cluster_lock_file,
                    label="Bridge Stage 2 epoch {}".format(epoch)):
                distance_aerial = baseline.compute_jaccard_distance(
                    features_aerial, k1=args.k1, k2=args.k2,
                    search_option=3)
                labels_aerial, num_aerial = _adaptive_cluster(
                    cluster_aerial, distance_aerial, "Aerial")
                del distance_aerial

                distance_ground = baseline.compute_jaccard_distance(
                    features_ground, k1=args.k1, k2=args.k2,
                    search_option=3)
                labels_ground, num_ground = _adaptive_cluster(
                    cluster_ground, distance_ground, "Ground")
                del distance_ground

                all_modalities = None
                if args.dataset_key == "agreidv2":
                    all_modalities = ([1] * len(ground_train)
                                      + [2] * len(aerial_train))
                distance_all = baseline.compute_modal_invariant_jaccard_distance(
                    features_all, k1=40, k2=32, file=all_train,
                    search_option=3, modalities=all_modalities)
                dump_cluster_distance_if_requested(
                    distance_all, epoch, args.dump_cluster_dist_epoch,
                    args.logs_dir)
                labels_all, num_all = _adaptive_cluster(
                    cluster_all, distance_all, "ALL")
                del distance_all

        memory_aerial, centers_aerial, num_aerial = _create_hard_memory(
            model, args, labels_aerial, features_aerial,
            features_aerial_ema)
        memory_ground, centers_ground, num_ground = _create_hard_memory(
            model, args, labels_ground, features_ground,
            features_ground_ema)
        modality_is_aerial = ([False] * len(ground_train)
                              + [True] * len(aerial_train))
        centers_all = generate_joint_cluster_features(
            labels_all, num_all, features_all, modality_is_aerial)
        memory_all = ClusterMemory(
            model.module.num_features, num_all, temp=args.temp,
            momentum=args.momentum, mode="CMhybrid", smooth=args.smooth,
            num_instances=args.num_instances).cuda()
        memory_all.features = F.normalize(
            centers_all.repeat(2, 1), dim=1).cuda()
        trainer.memory_ir = memory_aerial
        trainer.memory_rgb = memory_ground
        trainer.memory_all = memory_all

        scores = _compute_epoch_bridge_scores(
            args, features_aerial, features_ground, csv_scores, epoch)
        near_mask, far_mask = select_near_members(
            labels_aerial, scores, ratio=args.bridge_ratio)
        if abs(args.bridge_ratio - 1.0) <= 1e-12:
            # Exact baseline control: extraction averages original/flip
            # embeddings, so individual rows are not guaranteed unit norm.
            # Reuse the baseline full-cluster centers instead of relying on
            # normalized-member averaging to be numerically equivalent.
            centers_bridge = centers_aerial
        else:
            centers_bridge = bridge_centers(
                features_aerial, labels_aerial, near_mask)
        selection = bridge_selection_stats(
            labels_aerial, scores, near_mask)

        # source=ground, target=aerial preserves baseline r2i/i2r direction.
        r2i, i2r, bridge_bgm_stats = _match_centers(
            args, centers_ground, centers_bridge)
        full_r2i, full_i2r, full_bgm_stats = _match_centers(
            args, centers_ground, centers_aerial)
        difference = matching_difference_stats(
            i2r, full_i2r, cluster_ids=range(num_aerial))
        print("==> Bridge clusters epoch {}: Aerial={}, Ground={}, ALL={}".format(
            epoch, num_aerial, num_ground, num_all))
        print("Bridge selection: near={}, far={}, outliers={}, "
              "near_score_mean={:.6f}, far_score_mean={:.6f}".format(
                  selection["near_images"], selection["far_images"],
                  selection["outliers"], selection["near_score_mean"],
                  selection["far_score_mean"]))
        print("Bridge-center BGM: {}".format(bridge_bgm_stats))
        print("Full-center diagnostic BGM: {}".format(full_bgm_stats))
        print("Bridge/full different aerial matches: {}/{} ({:.2%})".format(
            difference["different_clusters"], difference["total_clusters"],
            difference["different_ratio"]))
        if args.log_gt_match_accuracy:
            aerial_names = [sample[0] for sample in aerial_train]
            ground_names = [sample[0] for sample in ground_train]
            _format_accuracy("bridge centers", gt_match_accuracy(
                labels_aerial, aerial_names, labels_ground, ground_names,
                i2r))
            _format_accuracy("full centers", gt_match_accuracy(
                labels_aerial, aerial_names, labels_ground, ground_names,
                full_i2r))

        if args.confident_cross:
            if args.bgm_fallback == "pclhd" and num_ground < num_aerial:
                print(
                    "WARNING: ground clusters ({}) < aerial clusters ({}); "
                    "cross-view loss is disabled this epoch. "
                    "Use --bgm-fallback nearest.".format(
                        num_ground, num_aerial))
            retain_ratio = cross_retain_ratio(
                epoch, args.cross_retain_start, args.cross_retain_end,
                args.cross_retain_ramp_epochs)
            i2r_confidence, r2i_confidence = \
                cross_view_mapping_confidences(
                    centers_ground, centers_bridge, i2r, r2i,
                    mode=args.cross_confidence)
            retained_i2r, removed_i2r, i2r_stats = \
                retain_mapping_by_confidence(
                    i2r, i2r_confidence, retain_ratio, num_aerial)
            retained_r2i, removed_r2i, r2i_stats = \
                retain_mapping_by_confidence(
                    r2i, r2i_confidence, retain_ratio, num_ground)
            print("Confidence-aware cross mapping: epoch={}, rho={:.6f}, "
                  "mode={}".format(
                      epoch, retain_ratio, args.cross_confidence))
            print("  i2r retained={}/{}, available={}, "
                  "mean_confidence={:.6f}".format(
                      i2r_stats["retained"], i2r_stats["total_clusters"],
                      i2r_stats["available"],
                      i2r_stats["mean_confidence"]))
            print("  r2i retained={}/{}, available={}, "
                  "mean_confidence={:.6f}".format(
                      r2i_stats["retained"], r2i_stats["total_clusters"],
                      r2i_stats["available"],
                      r2i_stats["mean_confidence"]))
            if args.log_gt_match_accuracy:
                _format_accuracy("retained i2r", gt_match_accuracy(
                    labels_aerial, aerial_names,
                    labels_ground, ground_names, retained_i2r))
                _format_accuracy("removed i2r", gt_match_accuracy(
                    labels_aerial, aerial_names,
                    labels_ground, ground_names, removed_i2r))
                _format_accuracy("retained r2i", gt_match_accuracy(
                    labels_ground, ground_names,
                    labels_aerial, aerial_names, retained_r2i))
                _format_accuracy("removed r2i", gt_match_accuracy(
                    labels_ground, ground_names,
                    labels_aerial, aerial_names, removed_r2i))
            i2r = retained_i2r
            r2i = retained_r2i

        pseudo_aerial = [
            (fname, int(label), cid)
            for (fname, _, cid), label in zip(aerial_train, labels_aerial)
            if int(label) != -1]
        pseudo_ground = [
            (fname, int(label), cid)
            for (fname, _, cid), label in zip(ground_train, labels_ground)
            if int(label) != -1]
        valid_joint_clusters = joint_cross_modal_flags(
            labels_all, modality_is_aerial)
        joint_ground = []
        joint_aerial = []
        for index, ((fname, _, cid), raw_label) in enumerate(
                zip(all_train, labels_all)):
            label = int(raw_label)
            if label not in valid_joint_clusters:
                continue
            if index < len(ground_train):
                joint_ground.append((fname, label, cid))
            else:
                joint_aerial.append((fname, label, cid))

        for name, pseudo_set in (
                ("Aerial", pseudo_aerial), ("Ground", pseudo_ground),
                ("ALL Aerial", joint_aerial),
                ("ALL Ground", joint_ground)):
            _require_nonempty(name, pseudo_set)

        ground_transform, ground_aug_transform, aerial_transform = \
            _build_transforms(args)
        ir_loader_builder = (
            get_train_loader_ir_geometry
            if geometry_aspect_jitter_enabled(args, 2) else
            baseline.get_train_loader_ir)
        ground_loader_builder = (
            get_train_loader_color_geometry
            if geometry_aspect_jitter_enabled(args, 2) else
            baseline.get_train_loader_color)
        loader_aerial = ir_loader_builder(
            args, dataset_ir, args.height, args.width, args.batch_size,
            args.workers, args.num_instances, iters,
            trainset=pseudo_aerial, no_cam=args.no_cam,
            train_transformer=aerial_transform,
            resolution_process=resolution_augmenters["aerial"])
        loader_ground = ground_loader_builder(
            args, dataset_ground, args.height, args.width, args.batch_size,
            args.workers, args.num_instances, iters,
            trainset=pseudo_ground, no_cam=args.no_cam,
            train_transformer=ground_transform,
            train_transformer1=ground_aug_transform,
            resolution_process=resolution_augmenters["ground"])
        loader_all_aerial = ir_loader_builder(
            args, dataset_ir, args.height, args.width, args.batch_size,
            args.workers, args.num_instances, iters,
            trainset=joint_aerial, no_cam=args.no_cam,
            train_transformer=aerial_transform,
            resolution_process=resolution_augmenters["aerial"])
        loader_all_ground = ground_loader_builder(
            args, dataset_ground, args.height, args.width, args.batch_size,
            args.workers, args.num_instances, iters,
            trainset=joint_ground, no_cam=args.no_cam,
            train_transformer=ground_transform,
            train_transformer1=ground_aug_transform,
            resolution_process=resolution_augmenters["ground"])
        loaders = (loader_aerial, loader_ground,
                   loader_all_aerial, loader_all_ground)
        for loader in loaders:
            loader.new_epoch()

        trainer.train(
            epoch, *loaders, optimizer, print_freq=args.print_freq,
            train_iters=len(loader_aerial), i2r=i2r, r2i=r2i)
        print_resolution_stats(2, epoch, resolution_augmenters)
        best_r1, best_map, best_epoch = _evaluate(
            model_ema, args, data_dir, epoch,
            best_r1, best_map, best_epoch)
        lr_scheduler.step()
        print("Stage 2 bridge epoch {} clustering eps used: "
              "Aerial={:.3f}, Ground={:.3f}, ALL={:.3f}".format(
                  epoch, cluster_aerial.last_used_eps,
                  cluster_ground.last_used_eps, cluster_all.last_used_eps))

    print("Total running time: {}".format(
        timedelta(seconds=time.monotonic() - start_time)))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Internal AG-ReID bridge-matching Stage 2")
    parser.add_argument("-d", "--dataset", type=str, default="agreid_ir")
    parser.add_argument(
        "--dataset-key", choices=("agreid", "agreidv2"), default="agreid")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("--test-batch", type=int, default=64)
    parser.add_argument("-j", "--workers", type=int, default=8)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--num-instances", type=int, default=16)
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--eps-all", type=float, default=None)
    parser.add_argument(
        "--dbscan-backend", choices=("sklearn", "fast"),
        default="sklearn")
    parser.add_argument(
        "--dump-cluster-dist-epoch", type=int, default=-1)
    parser.add_argument("--cluster-lock-file", type=str, default="")
    parser.add_argument("--eps-gap", type=float, default=0.02)
    parser.add_argument("--k1", type=int, default=30)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("-a", "--arch", type=str, default="agw")
    parser.add_argument("--features", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--momentum", type=float, default=0.2)
    parser.add_argument("-mb", "--memorybank", type=str, default="CMhard")
    parser.add_argument("--smooth", type=float, default=0)
    parser.add_argument("--lr", type=float, default=0.00035)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--eval-step", type=int, default=1)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.05)
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument("--data-dir", type=str,
                        default=osp.join(working_dir, "data"))
    parser.add_argument("--logs-dir", type=str,
                        default=osp.join(working_dir, "logs"))
    parser.add_argument("--stage1-log-name", type=str, default="")
    parser.add_argument("--pooling-type", type=str, default="gem")
    parser.add_argument("--use-hard", action="store_true")
    parser.add_argument("--no-cam", action="store_true")
    parser.add_argument("--stage2-only", action="store_true")
    parser.add_argument("--stage1-checkpoint", type=str, required=True)
    parser.add_argument("--stage2-log-name", type=str,
                        default=DEFAULT_STAGE2_LOG_NAME)
    parser.add_argument("--allow-log-overwrite", action="store_true")
    parser.add_argument("--debug-nonfinite", action="store_true")
    parser.add_argument(
        "--channel-adap-gray", choices=("off",), default="off")
    parser.add_argument(
        "--bridge-score", choices=BRIDGE_SCORE_MODES,
        default="ground_knn")
    parser.add_argument("--bridge-topk", type=int, default=10)
    parser.add_argument("--bridge-ratio", type=float, default=0.5)
    parser.add_argument(
        "--bgm-fallback", choices=BGM_FALLBACK_MODES, default="pclhd")
    parser.add_argument("--confident-cross", action="store_true")
    parser.add_argument("--cross-retain-start", type=float, default=0.4)
    parser.add_argument("--cross-retain-end", type=float, default=1.0)
    parser.add_argument(
        "--cross-retain-ramp-epochs", type=int, default=20)
    parser.add_argument(
        "--cross-confidence", choices=CROSS_CONFIDENCE_MODES,
        default="margin")
    parser.add_argument("--view-score-file", type=str, default="")
    parser.add_argument(
        "--view-score-path-mode", choices=VIEW_PATH_MODES,
        default="basename")
    parser.add_argument("--log-gt-match-accuracy", action="store_true")
    parser.set_defaults(
        use_agva_ir=False, agva_orig_root="", agva_ir_root="",
        agva_ir_prob=0.0)
    parser.add_argument("--dynamic-agva-mask-dir", default="")
    parser.add_argument("--dynamic-agva-image-root", default="")
    parser.set_defaults(use_dynamic_agva=False)
    add_aspect_jitter_arguments(parser)
    add_resolution_augment_arguments(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(
        sys.argv[1:] if argv is None else argv)
    validate_resolution_augment_args(args)
    validate_aspect_jitter_args(args)
    args.eps_all = resolve_eps_all(args)
    args.cluster_lock_file = resolve_cluster_lock_file(
        args.cluster_lock_file)
    args.stage2_log_name = append_resolution_log_suffix(
        args.stage2_log_name, args)
    args.stage2_log_name = append_stage2_eps_log_suffix(
        args.stage2_log_name, args.eps, args.eps_all)
    _validate_args(args)
    args.logs_root = osp.abspath(osp.expanduser(args.logs_dir))
    output_dir = get_stage2_output_directory(
        args.stage2_log_name, args.trial, logs_root=args.logs_root)
    print("Experiment: BRIDGE-MATCHING STAGE 2 (NO EMA LOSS)")
    print("Bridge score: {}".format(args.bridge_score))
    print("Bridge top-k: {}".format(args.bridge_topk))
    print("Bridge ratio: {}".format(args.bridge_ratio))
    print("BGM fallback: {}".format(args.bgm_fallback))
    print("Confidence-aware cross matching: {}".format(
        args.confident_cross))
    if args.confident_cross:
        print("Cross retention: start={}, end={}, ramp_epochs={}".format(
            args.cross_retain_start, args.cross_retain_end,
            args.cross_retain_ramp_epochs))
        print("Cross confidence: {}".format(args.cross_confidence))
    print("GT match diagnostics: {}".format(
        args.log_gt_match_accuracy))
    print("DBSCAN backend: {}".format(args.dbscan_backend))
    print("Cluster lock: {}".format(
        args.cluster_lock_file if args.cluster_lock_file else "disabled"))
    print("Stage 1: skipped; explicit checkpoint required")
    print("Stage 2 log name: {}".format(args.stage2_log_name))
    print("Stage 2 output directory: {}".format(output_dir))
    check_stage2_output_directory(
        output_dir, allow_overwrite=args.allow_log_overwrite)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker_stage2(args)


if __name__ == "__main__":
    main()
