#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline evaluation of AG-ReID aerial-to-ground cluster matching.

This tool never trains or writes checkpoints.  For each supplied checkpoint it
extracts the same normalized training features and runs the same three Stage-2
clusterings as the baseline engine, then compares several matching strategies
using filename-derived identities strictly as evaluation labels.
"""

from __future__ import absolute_import, print_function

import argparse
import collections
import csv
import json
import math
import os
import os.path as osp
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.backends import cudnn
from torch.nn import functional as F


ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import _train_agreid_engine as baseline
from agreid_identity import parse_agreid_name
from clustercontrast.bridge_matching import (
    bridge_centers, ground_knn_scores, gt_match_accuracy, pclhd_bgm,
    select_near_members)
from clustercontrast.utils.cluster_lock import (
    cluster_lock, resolve_cluster_lock_file)


CSV_FIELDS = (
    "checkpoint", "method", "parameters", "pair_acc", "image_acc",
    "coverage", "aerial_purity", "ground_purity", "all_purity",
    "matched_pairs", "clustered_aerial_images")
BRIDGE_CHUNK_SIZE = 1024
CONFIDENCE_RETAIN_RATIOS = (0.2, 0.4, 0.6, 0.8, 1.0)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Offline AG-ReID cross-view cluster-matching analysis")
    parser.add_argument(
        "--dataset-key", choices=("agreid", "agreidv2"), required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--checkpoint", action="append", nargs="+", required=True,
        help=("one or more checkpoint paths; the option may also be "
              "repeated"))
    parser.add_argument("--eps", type=float, default=0.6)
    parser.add_argument(
        "--dbscan-backend", choices=("sklearn", "fast"), default="fast")
    parser.add_argument(
        "--cluster-lock-file", default="",
        help="shared flock path; empty uses PCLHD_CLUSTER_LOCK or no lock")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--bridge-ratios", type=float, nargs="+",
        default=(0.25, 0.5, 0.75))
    parser.add_argument(
        "--bridge-topks", type=int, nargs="+", default=(5, 10, 20))
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=(0.5, 1.0, 2.0))
    parser.add_argument(
        "--confidence", choices=("similarity", "margin"), default="margin",
        help="score used to retain the top 20%%, 40%%, ..., 100%% matches")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--test-batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--arch", default="agw")
    parser.add_argument("--features", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pooling-type", default="gem")
    parser.set_defaults(k1=30, k2=6, eps_all=None)
    return parser


def _validate_args(args, parser):
    if args.eps <= 0:
        parser.error("--eps must be positive")
    if not args.bridge_ratios or any(
            ratio <= 0.0 or ratio > 1.0 for ratio in args.bridge_ratios):
        parser.error("--bridge-ratios values must be in (0, 1]")
    if not args.bridge_topks or any(topk < 1 for topk in args.bridge_topks):
        parser.error("--bridge-topks values must be at least 1")
    if not args.lambdas or any(value < 0.0 for value in args.lambdas):
        parser.error("--lambdas values must be non-negative")
    if args.test_batch < 1 or args.workers < 0:
        parser.error("--test-batch must be positive and --workers non-negative")
    for checkpoint in args.checkpoint:
        if not osp.isfile(osp.abspath(osp.expanduser(checkpoint))):
            parser.error("checkpoint not found: {}".format(checkpoint))


def _extract_ordered_features(model, dataset, trainset, args, mode):
    loader = baseline.get_test_loader(
        dataset, args.height, args.width, args.test_batch, args.workers,
        testset=trainset)
    feature_map, _ = baseline.extract_features(
        model, loader, print_freq=50, mode=mode)
    return torch.cat([
        feature_map[fname].unsqueeze(0) for fname, _, _ in trainset
    ], dim=0)


def _adaptive_cluster(clusterer, distance, name, max_attempts=10):
    attempts = 0
    while attempts < max_attempts:
        used_eps = clusterer.eps
        labels = clusterer.fit_predict(distance)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if num_clusters > 0:
            print("{} clusters: {} (eps={:.3f})".format(
                name, num_clusters, used_eps))
            return np.asarray(labels, dtype=np.int64), num_clusters
        clusterer.eps += 0.05
        print("{}: no clusters, increasing eps to {:.3f}".format(
            name, clusterer.eps))
        attempts += 1
    print("WARNING: {} clustering failed; assigning cluster 0".format(name))
    return np.zeros(len(distance), dtype=np.int64), 1


def _cluster_checkpoint(args, checkpoint, aerial_train, ground_train,
                        dataset_aerial, dataset_ground):
    print("\n========== Checkpoint: {} ==========".format(checkpoint))
    model, unused_model_ema = baseline.create_model(args)
    del unused_model_ema
    checkpoint_data = baseline.load_checkpoint(checkpoint)
    if "state_dict" not in checkpoint_data:
        raise KeyError("checkpoint has no state_dict: {}".format(checkpoint))
    model.load_state_dict(checkpoint_data["state_dict"])
    model.eval()

    with torch.no_grad():
        print("==> Extract ground training features")
        features_ground = _extract_ordered_features(
            model, dataset_ground, ground_train, args, mode=1)
        print("==> Extract aerial training features")
        features_aerial = _extract_ordered_features(
            model, dataset_aerial, aerial_train, args, mode=2)
    del model, checkpoint_data
    torch.cuda.empty_cache()

    features_all = torch.cat((features_ground, features_aerial), dim=0)
    all_train = ground_train + aerial_train
    clusterers = baseline.build_agreid_stage2_clusterers(args)
    with cluster_lock(
            args.cluster_lock_file,
            label="offline matching {}".format(osp.basename(checkpoint))):
        distance_aerial = baseline.compute_jaccard_distance(
            features_aerial, k1=args.k1, k2=args.k2, search_option=3)
        labels_aerial, num_aerial = _adaptive_cluster(
            clusterers["ir"], distance_aerial, "Aerial")
        del distance_aerial

        distance_ground = baseline.compute_jaccard_distance(
            features_ground, k1=args.k1, k2=args.k2, search_option=3)
        labels_ground, num_ground = _adaptive_cluster(
            clusterers["rgb"], distance_ground, "Ground")
        del distance_ground

        modalities = None
        if args.dataset_key == "agreidv2":
            modalities = ([1] * len(ground_train)
                          + [2] * len(aerial_train))
        distance_all = baseline.compute_modal_invariant_jaccard_distance(
            features_all, k1=40, k2=32, file=all_train,
            search_option=3, modalities=modalities)
        labels_all, num_all = _adaptive_cluster(
            clusterers["all"], distance_all, "ALL")
        del distance_all

    return {
        "features_aerial": features_aerial,
        "features_ground": features_ground,
        "labels_aerial": labels_aerial,
        "labels_ground": labels_ground,
        "labels_all": labels_all,
        "num_aerial": num_aerial,
        "num_ground": num_ground,
        "num_all": num_all,
    }


@torch.no_grad()
def _cluster_centers(features, labels, num_clusters):
    labels_tensor = torch.as_tensor(
        labels, dtype=torch.long, device=features.device)
    return torch.stack([
        features[labels_tensor == cluster_id].mean(dim=0)
        for cluster_id in range(num_clusters)
    ], dim=0)


def _cluster_majorities(labels, filenames):
    values = collections.defaultdict(list)
    for raw_label, filename in zip(labels, filenames):
        label = int(raw_label)
        if label != -1:
            values[label].append(parse_agreid_name(filename).pid)
    majorities = {}
    for cluster_id, identities in values.items():
        counts = collections.Counter(identities)
        majorities[cluster_id] = sorted(
            counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return majorities


def _cluster_purity(labels, filenames):
    values = collections.defaultdict(list)
    for raw_label, filename in zip(labels, filenames):
        label = int(raw_label)
        if label != -1:
            values[label].append(parse_agreid_name(filename).pid)
    purities = []
    for identities in values.values():
        majority = max(collections.Counter(identities).values())
        purities.append(float(majority) / float(len(identities)))
    return float(np.mean(purities)) if purities else float("nan")


def _evaluate_mapping(mapping, labels_aerial, aerial_names, labels_ground,
                      ground_names):
    pair_stats = gt_match_accuracy(
        labels_aerial, aerial_names, labels_ground, ground_names, mapping)
    ground_majority = _cluster_majorities(labels_ground, ground_names)
    clustered_images = 0
    correct_images = 0
    aerial_clusters = {
        int(label) for label in labels_aerial if int(label) != -1}
    for raw_label, filename in zip(labels_aerial, aerial_names):
        aerial_cluster = int(raw_label)
        if aerial_cluster == -1:
            continue
        clustered_images += 1
        ground_cluster = mapping.get(aerial_cluster)
        if ground_cluster is None or int(ground_cluster) not in ground_majority:
            continue
        correct_images += int(
            ground_majority[int(ground_cluster)]
            == parse_agreid_name(filename).pid)
    image_acc = (float(correct_images) / float(clustered_images)
                 if clustered_images else float("nan"))
    matched_clusters = sum(
        cluster_id in mapping for cluster_id in aerial_clusters)
    coverage = (float(matched_clusters) / float(len(aerial_clusters))
                if aerial_clusters else 0.0)
    return {
        "pair_acc": pair_stats["accuracy"],
        "image_acc": image_acc,
        "coverage": coverage,
        "matched_pairs": pair_stats["evaluated_pairs"],
        "clustered_aerial_images": clustered_images,
    }


def _cooccurrence_similarity(labels_ground, labels_aerial, labels_all,
                             num_ground, num_aerial):
    """Return Dg^-1/2 C Da^-1/2 from joint-cluster pair counts."""
    labels_ground = np.asarray(labels_ground, dtype=np.int64)
    labels_aerial = np.asarray(labels_aerial, dtype=np.int64)
    labels_all = np.asarray(labels_all, dtype=np.int64)
    expected = len(labels_ground) + len(labels_aerial)
    if len(labels_all) != expected:
        raise ValueError("ALL labels do not match ground+aerial rows")
    matrix = np.zeros((num_ground, num_aerial), dtype=np.float64)
    all_ground = labels_all[:len(labels_ground)]
    all_aerial = labels_all[len(labels_ground):]
    joint_ids = sorted({
        int(value) for value in labels_all.tolist() if int(value) != -1})
    for joint_id in joint_ids:
        ground_counts = collections.Counter(
            int(value) for value in labels_ground[all_ground == joint_id]
            if int(value) != -1)
        aerial_counts = collections.Counter(
            int(value) for value in labels_aerial[all_aerial == joint_id]
            if int(value) != -1)
        for ground_cluster, ground_count in ground_counts.items():
            for aerial_cluster, aerial_count in aerial_counts.items():
                matrix[ground_cluster, aerial_cluster] += (
                    ground_count * aerial_count)
    row_sum = matrix.sum(axis=1, keepdims=True)
    col_sum = matrix.sum(axis=0, keepdims=True)
    denominator = np.sqrt(row_sum * col_sum)
    normalized = np.divide(
        matrix, denominator, out=np.zeros_like(matrix),
        where=denominator > 0)
    return torch.from_numpy(normalized).to(dtype=torch.float32)


def _pclhd_similarity_bgm(similarity):
    """Use baseline PGM padding/fallback with cost=-similarity."""
    if similarity.ndim != 2:
        raise ValueError("matching similarity must be two-dimensional")
    num_ground, num_aerial = map(int, similarity.shape)
    if num_ground == 0 or num_aerial == 0:
        raise ValueError("matching requires non-empty cluster sets")
    r2i = {}
    i2r = {}
    if num_ground < num_aerial:
        return r2i, i2r
    cost = -similarity.detach().cpu()
    padding = torch.zeros(num_ground, num_ground - num_aerial)
    cost = torch.cat((cost, padding), dim=1)
    unmatched_rows = []
    row_ind, col_ind = linear_sum_assignment(cost)
    for row, column in zip(row_ind, col_ind):
        if column < num_aerial:
            r2i[int(row)] = int(column)
            i2r[int(column)] = int(row)
        else:
            unmatched_rows.append(int(row))
    unmatched_cost = cost[unmatched_rows, :num_aerial]
    fallback_rows, fallback_columns = linear_sum_assignment(unmatched_cost)
    for row, column in zip(fallback_rows, fallback_columns):
        r2i[unmatched_rows[int(row)]] = int(column)
    return r2i, i2r


def _match_confidence(mapping, similarity, mode):
    confidence = {}
    for aerial_cluster, ground_cluster in mapping.items():
        column = similarity[:, int(aerial_cluster)]
        selected = float(column[int(ground_cluster)].item())
        if mode == "similarity" or column.numel() == 1:
            confidence[int(aerial_cluster)] = selected
            continue
        alternatives = torch.cat((
            column[:int(ground_cluster)],
            column[int(ground_cluster) + 1:]))
        confidence[int(aerial_cluster)] = (
            selected - float(alternatives.max().item()))
    return confidence


def _retain_mapping(mapping, confidence, ratio):
    if not mapping:
        return {}
    keep = max(1, int(math.ceil(float(ratio) * len(mapping))))
    ordered = sorted(
        mapping, key=lambda cluster_id: (
            -confidence[int(cluster_id)], int(cluster_id)))
    selected = set(ordered[:keep])
    return {
        int(cluster_id): int(ground_id)
        for cluster_id, ground_id in mapping.items()
        if int(cluster_id) in selected
    }


def _parameters(**values):
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _result_row(checkpoint, method, parameters, metrics, purities):
    return {
        "checkpoint": checkpoint,
        "method": method,
        "parameters": parameters,
        "pair_acc": metrics["pair_acc"],
        "image_acc": metrics["image_acc"],
        "coverage": metrics["coverage"],
        "aerial_purity": purities["aerial"],
        "ground_purity": purities["ground"],
        "all_purity": purities["all"],
        "matched_pairs": metrics["matched_pairs"],
        "clustered_aerial_images": metrics["clustered_aerial_images"],
    }


def _analyze_checkpoint(args, checkpoint, clustered, aerial_train,
                        ground_train):
    features_aerial = clustered["features_aerial"]
    features_ground = clustered["features_ground"]
    labels_aerial = clustered["labels_aerial"]
    labels_ground = clustered["labels_ground"]
    labels_all = clustered["labels_all"]
    num_aerial = clustered["num_aerial"]
    num_ground = clustered["num_ground"]
    aerial_names = [sample[0] for sample in aerial_train]
    ground_names = [sample[0] for sample in ground_train]
    all_names = ground_names + aerial_names
    purities = {
        "aerial": _cluster_purity(labels_aerial, aerial_names),
        "ground": _cluster_purity(labels_ground, ground_names),
        "all": _cluster_purity(labels_all, all_names),
    }
    print("Cluster purity: aerial={:.4f}, ground={:.4f}, ALL={:.4f}".format(
        purities["aerial"], purities["ground"], purities["all"]))

    centers_aerial = _cluster_centers(
        features_aerial, labels_aerial, num_aerial)
    centers_ground = _cluster_centers(
        features_ground, labels_ground, num_ground)
    cosine_similarity = F.normalize(centers_ground, dim=1).mm(
        F.normalize(centers_aerial, dim=1).t()).cpu()
    cooccur_similarity = _cooccurrence_similarity(
        labels_ground, labels_aerial, labels_all, num_ground, num_aerial)

    rows = []
    confidence_candidates = []
    _, full_i2r, _ = pclhd_bgm(centers_ground, centers_aerial)
    full_metrics = _evaluate_mapping(
        full_i2r, labels_aerial, aerial_names, labels_ground, ground_names)
    rows.append(_result_row(
        checkpoint, "full", _parameters(bgm="pclhd"),
        full_metrics, purities))
    confidence_candidates.append({
        "method": "full", "parameters": {"bgm": "pclhd"},
        "mapping": full_i2r, "similarity": cosine_similarity,
        "metrics": full_metrics,
    })

    print("Computing ground_knn scores on GPU in {}-row chunks".format(
        BRIDGE_CHUNK_SIZE))
    aerial_cuda = features_aerial.cuda(non_blocking=True)
    ground_cuda = features_ground.cuda(non_blocking=True)
    for topk in args.bridge_topks:
        scores = ground_knn_scores(
            aerial_cuda, ground_cuda, topk=topk,
            chunk_size=BRIDGE_CHUNK_SIZE).cpu()
        for ratio in args.bridge_ratios:
            near_mask, _ = select_near_members(
                labels_aerial, scores, ratio=ratio)
            centers_bridge = bridge_centers(
                features_aerial, labels_aerial, near_mask)
            _, bridge_i2r, _ = pclhd_bgm(
                centers_ground, centers_bridge)
            metrics = _evaluate_mapping(
                bridge_i2r, labels_aerial, aerial_names,
                labels_ground, ground_names)
            rows.append(_result_row(
                checkpoint, "bridge",
                _parameters(topk=int(topk), ratio=float(ratio), bgm="pclhd"),
                metrics, purities))
    del aerial_cuda, ground_cuda
    torch.cuda.empty_cache()

    _, cooccur_i2r = _pclhd_similarity_bgm(cooccur_similarity)
    cooccur_metrics = _evaluate_mapping(
        cooccur_i2r, labels_aerial, aerial_names, labels_ground, ground_names)
    rows.append(_result_row(
        checkpoint, "cooccur",
        _parameters(normalization="sqrt_row_col", cost="negative"),
        cooccur_metrics, purities))

    for lambda_value in args.lambdas:
        combined_similarity = (
            cosine_similarity + float(lambda_value) * cooccur_similarity)
        _, combined_i2r = _pclhd_similarity_bgm(combined_similarity)
        combined_metrics = _evaluate_mapping(
            combined_i2r, labels_aerial, aerial_names,
            labels_ground, ground_names)
        params = {"lambda": float(lambda_value), "cost": "negative"}
        rows.append(_result_row(
            checkpoint, "combined", _parameters(**params),
            combined_metrics, purities))
        confidence_candidates.append({
            "method": "combined", "parameters": params,
            "mapping": combined_i2r,
            "similarity": combined_similarity,
            "metrics": combined_metrics,
        })

    best = max(confidence_candidates, key=lambda item: (
        _sortable(item["metrics"]["image_acc"]),
        _sortable(item["metrics"]["pair_acc"]),
        item["metrics"]["coverage"]))
    confidence = _match_confidence(
        best["mapping"], best["similarity"], args.confidence)
    print("Confidence filtering base: {} {}".format(
        best["method"], _parameters(**best["parameters"])))
    for retain_ratio in CONFIDENCE_RETAIN_RATIOS:
        retained = _retain_mapping(
            best["mapping"], confidence, retain_ratio)
        metrics = _evaluate_mapping(
            retained, labels_aerial, aerial_names,
            labels_ground, ground_names)
        params = dict(best["parameters"])
        params.update({
            "base_method": best["method"],
            "confidence": args.confidence,
            "retain": retain_ratio,
        })
        rows.append(_result_row(
            checkpoint, "confidence", _parameters(**params),
            metrics, purities))
    return rows


def _sortable(value):
    value = float(value)
    return value if math.isfinite(value) else float("-inf")


def _write_csv(path, rows):
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_top_rows(rows, limit=10):
    ordered = sorted(rows, key=lambda row: (
        -_sortable(row["image_acc"]),
        -_sortable(row["pair_acc"]),
        -_sortable(row["coverage"]),
        row["checkpoint"], row["method"], row["parameters"]))
    print("\n========== Top {} by image_acc ==========".format(
        min(limit, len(ordered))))
    print("rank  image_acc  pair_acc  coverage  checkpoint  method  parameters")
    for rank, row in enumerate(ordered[:limit], start=1):
        checkpoint_name = osp.basename(osp.dirname(row["checkpoint"])) or \
            osp.basename(row["checkpoint"])
        print("{:>4}  {:>9.4f}  {:>8.4f}  {:>8.4f}  {}  {}  {}".format(
            rank, row["image_acc"], row["pair_acc"], row["coverage"],
            checkpoint_name, row["method"], row["parameters"]))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    args.data_dir = osp.abspath(osp.expanduser(args.data_dir))
    args.output_csv = osp.abspath(osp.expanduser(args.output_csv))
    args.checkpoint = [
        osp.abspath(osp.expanduser(path))
        for group in args.checkpoint for path in group]
    args.cluster_lock_file = resolve_cluster_lock_file(
        args.cluster_lock_file)
    _validate_args(args, parser)
    cudnn.benchmark = True

    print("Dataset: {}".format(args.dataset_key))
    print("Data root: {}".format(args.data_dir))
    print("Checkpoints: {}".format(len(args.checkpoint)))
    print("Clustering: eps={}, eps_all={}, backend={}, k1={}, k2={}".format(
        args.eps, args.eps, args.dbscan_backend, args.k1, args.k2))
    print("Cluster lock: {}".format(
        args.cluster_lock_file if args.cluster_lock_file else "disabled"))
    print("Bridge ratios: {}".format(list(args.bridge_ratios)))
    print("Bridge top-k values: {}".format(list(args.bridge_topks)))
    print("Combined lambdas: {}".format(list(args.lambdas)))
    print("Confidence: {}".format(args.confidence))

    aerial_name, ground_name = baseline.get_training_dataset_names(
        args.dataset_key)
    dataset_aerial = baseline.get_data(
        aerial_name, args.data_dir, trial=args.trial)
    dataset_ground = baseline.get_data(
        ground_name, args.data_dir, trial=args.trial)
    aerial_train = sorted(dataset_aerial.train)
    ground_train = sorted(dataset_ground.train)

    rows = []
    for checkpoint in args.checkpoint:
        clustered = _cluster_checkpoint(
            args, checkpoint, aerial_train, ground_train,
            dataset_aerial, dataset_ground)
        rows.extend(_analyze_checkpoint(
            args, checkpoint, clustered, aerial_train, ground_train))
        del clustered
        torch.cuda.empty_cache()

    _write_csv(args.output_csv, rows)
    _print_top_rows(rows)
    print("\nWrote {} rows to {}".format(len(rows), args.output_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
