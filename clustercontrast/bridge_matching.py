# -*- coding: utf-8 -*-
"""Pure helpers for cluster-level bridge matching.

The training engine owns data loading, CSV I/O and logging.  Functions in
this module are deterministic transformations of their inputs so they can be
covered by CPU-only tests.
"""

from __future__ import absolute_import

import collections
import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from agreid_identity import parse_agreid_name


BRIDGE_SCORE_MODES = ("ground_knn", "view_csv", "random")
BGM_FALLBACK_MODES = ("pclhd", "nearest")
CROSS_CONFIDENCE_MODES = ("margin", "similarity")


def cross_retain_ratio(epoch, start=0.4, end=1.0, ramp_epochs=20):
    """Return the linear confidence-retention schedule for one epoch."""
    epoch = int(epoch)
    ramp_epochs = int(ramp_epochs)
    start = float(start)
    end = float(end)
    if epoch < 0:
        raise ValueError("cross-retain epoch cannot be negative")
    if ramp_epochs < 1:
        raise ValueError("cross-retain ramp epochs must be at least 1")
    if not 0.0 < start <= end <= 1.0:
        raise ValueError("cross-retain values must satisfy 0 < start <= end <= 1")
    progress = min(1.0, float(epoch) / float(ramp_epochs))
    return start + (end - start) * progress


@torch.no_grad()
def cross_view_mapping_confidences(ground_centers, aerial_centers,
                                   i2r, r2i, mode="margin"):
    """Score aerial->ground and ground->aerial mappings by cosine confidence."""
    if mode not in CROSS_CONFIDENCE_MODES:
        raise ValueError("unknown cross-confidence mode: {}".format(mode))
    if ground_centers.ndim != 2 or aerial_centers.ndim != 2:
        raise ValueError("cross-confidence centers must be two-dimensional")
    if ground_centers.size(1) != aerial_centers.size(1):
        raise ValueError("cross-confidence feature dimensions do not match")
    if ground_centers.size(0) == 0 or aerial_centers.size(0) == 0:
        raise ValueError("cross-confidence requires non-empty centers")
    ground = F.normalize(ground_centers, p=2, dim=1)
    aerial = F.normalize(aerial_centers, p=2, dim=1)
    similarity = ground.mm(aerial.t())

    def _score(vector, selected_index):
        selected_index = int(selected_index)
        if selected_index < 0 or selected_index >= vector.numel():
            raise ValueError("cross-confidence mapping index is out of range")
        selected = float(vector[selected_index].item())
        if mode == "similarity" or vector.numel() == 1:
            return selected
        alternatives = torch.cat(
            (vector[:selected_index], vector[selected_index + 1:]))
        return selected - float(alternatives.max().item())

    i2r_confidence = {}
    for aerial_cluster, ground_cluster in i2r.items():
        aerial_cluster = int(aerial_cluster)
        ground_cluster = int(ground_cluster)
        if aerial_cluster < 0 or aerial_cluster >= similarity.size(1):
            raise ValueError("i2r aerial cluster is out of range")
        i2r_confidence[aerial_cluster] = _score(
            similarity[:, aerial_cluster], ground_cluster)

    r2i_confidence = {}
    for ground_cluster, aerial_cluster in r2i.items():
        ground_cluster = int(ground_cluster)
        aerial_cluster = int(aerial_cluster)
        if ground_cluster < 0 or ground_cluster >= similarity.size(0):
            raise ValueError("r2i ground cluster is out of range")
        r2i_confidence[ground_cluster] = _score(
            similarity[ground_cluster, :], aerial_cluster)
    return i2r_confidence, r2i_confidence


def retain_mapping_by_confidence(mapping, confidence, ratio,
                                 total_clusters):
    """Keep the top ceil(ratio * total_clusters) mapping keys."""
    ratio = float(ratio)
    total_clusters = int(total_clusters)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("cross-retain ratio must be in (0, 1]")
    if total_clusters < 0:
        raise ValueError("cross-retain total clusters cannot be negative")
    normalized_mapping = {
        int(key): int(value) for key, value in mapping.items()}
    normalized_confidence = {
        int(key): float(value) for key, value in confidence.items()}
    missing = sorted(set(normalized_mapping) - set(normalized_confidence))
    if missing:
        raise ValueError(
            "missing confidence for mapping key {}".format(missing[0]))
    keep_count = min(
        len(normalized_mapping),
        int(math.ceil(ratio * float(total_clusters))))
    ordered = sorted(
        normalized_mapping,
        key=lambda key: (-normalized_confidence[key], key))
    keep_keys = set(ordered[:keep_count])
    retained = {
        key: value for key, value in normalized_mapping.items()
        if key in keep_keys}
    removed = {
        key: value for key, value in normalized_mapping.items()
        if key not in keep_keys}
    mean_confidence = (
        float(np.mean([normalized_confidence[key] for key in keep_keys]))
        if keep_keys else float("nan"))
    stats = {
        "retained": len(retained),
        "available": len(normalized_mapping),
        "total_clusters": total_clusters,
        "mean_confidence": mean_confidence,
    }
    return retained, removed, stats


@torch.no_grad()
def pclhd_bgm(ground_centers, aerial_centers):
    """Reproduce the baseline Stage-2 PGM mapping and fallback semantics."""
    if ground_centers.ndim != 2 or aerial_centers.ndim != 2:
        raise ValueError("BGM feature matrices must be two-dimensional")
    if ground_centers.size(0) == 0 or aerial_centers.size(0) == 0:
        raise ValueError("BGM requires at least one cluster on each side")
    if ground_centers.size(1) != aerial_centers.size(1):
        raise ValueError("BGM feature dimensions do not match")

    num_cluster_rgb = int(ground_centers.size(0))
    num_cluster_ir = int(aerial_centers.size(0))
    i2r = {}
    r2i = {}
    R = []
    bgm = False
    matched_pairs = 0
    fallback_pairs = 0
    if num_cluster_rgb >= num_cluster_ir:
        # This block intentionally mirrors the baseline Stage-2 PGM code.
        cluster_features_rgb = F.normalize(ground_centers, dim=1)
        cluster_features_ir = F.normalize(aerial_centers, dim=1)
        similarity = ((torch.mm(
            cluster_features_rgb, cluster_features_ir.T)) / 1).exp().cpu()
        dis_similarity = (1 / similarity)
        cost = dis_similarity / 1
        tmp = torch.zeros(
            dis_similarity.shape[0],
            dis_similarity.shape[0] - dis_similarity.shape[1])
        cost = torch.cat((cost, tmp), 1)
        unmatched_row = []
        row_ind, col_ind = linear_sum_assignment(cost)
        for idx, item in enumerate(row_ind):
            if col_ind[idx] < similarity.shape[1]:
                R.append((row_ind[idx], col_ind[idx]))
                r2i[row_ind[idx]] = col_ind[idx]
                i2r[col_ind[idx]] = row_ind[idx]
                matched_pairs += 1
            else:
                unmatched_row.append(row_ind[idx])
        if bgm is False:
            unmatched_cost = cost[unmatched_row][:, :dis_similarity.shape[1]]
            unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(
                unmatched_cost)
            for idx, item in enumerate(unmatched_row_ind):
                R.append((unmatched_row[idx], unmatched_col_ind[idx]))
                r2i[unmatched_row[idx]] = unmatched_col_ind[idx]
                fallback_pairs += 1

    stats = {
        "mode": "pclhd",
        "ground_clusters": num_cluster_rgb,
        "aerial_clusters": num_cluster_ir,
        "hungarian_pairs": matched_pairs,
        "ground_fallbacks": fallback_pairs,
        "unmapped_ground": num_cluster_rgb - len(r2i),
        "unmapped_aerial": num_cluster_ir - len(i2r),
        "cross_view_enabled": bool(r2i),
    }
    return r2i, i2r, stats


@torch.no_grad()
def ground_knn_scores(aerial_features, ground_features, topk=10,
                      chunk_size=1024):
    """Return mean top-k cosine similarity to ground, chunked by aerial row."""
    if aerial_features.ndim != 2 or ground_features.ndim != 2:
        raise ValueError("bridge feature matrices must be two-dimensional")
    if aerial_features.size(1) != ground_features.size(1):
        raise ValueError("aerial and ground feature dimensions do not match")
    if aerial_features.size(0) == 0 or ground_features.size(0) == 0:
        raise ValueError("ground_knn requires non-empty aerial and ground data")
    if int(topk) < 1:
        raise ValueError("bridge top-k must be at least 1")
    if int(chunk_size) < 1:
        raise ValueError("bridge chunk size must be at least 1")

    aerial = F.normalize(aerial_features, p=2, dim=1)
    ground = F.normalize(ground_features, p=2, dim=1)
    actual_k = min(int(topk), int(ground.size(0)))
    result = []
    for start in range(0, aerial.size(0), int(chunk_size)):
        chunk = aerial[start:start + int(chunk_size)]
        similarity = chunk.mm(ground.t())
        result.append(similarity.topk(actual_k, dim=1).values.mean(dim=1))
    return torch.cat(result, dim=0)


def random_bridge_scores(count, seed, epoch, device=None, dtype=torch.float32):
    """Return epoch-deterministic random bridge scores without global RNG use."""
    if int(count) < 0:
        raise ValueError("bridge score count cannot be negative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(epoch))
    scores = torch.rand(int(count), generator=generator, dtype=dtype)
    return scores if device is None else scores.to(device)


def bridge_scores(mode, aerial_features=None, ground_features=None, topk=10,
                  chunk_size=1024, csv_scores=None, seed=1, epoch=0):
    """Compute one score per aerial image from already prepared inputs."""
    if mode not in BRIDGE_SCORE_MODES:
        raise ValueError("unknown bridge score mode: {}".format(mode))
    if mode == "ground_knn":
        return ground_knn_scores(
            aerial_features, ground_features, topk=topk,
            chunk_size=chunk_size)
    if mode == "view_csv":
        if csv_scores is None:
            raise ValueError("view_csv bridge mode requires complete CSV scores")
        scores = torch.as_tensor(csv_scores, dtype=torch.float32)
        if aerial_features is not None:
            if scores.numel() != aerial_features.size(0):
                raise ValueError("CSV score count does not match aerial images")
            scores = scores.to(aerial_features.device)
        if not bool(torch.isfinite(scores).all().item()):
            raise ValueError("view_csv bridge scores must all be finite")
        return scores
    count = (len(csv_scores) if aerial_features is None
             else aerial_features.size(0))
    device = None if aerial_features is None else aerial_features.device
    return random_bridge_scores(count, seed, epoch, device=device)


def select_near_members(labels, scores, ratio=0.5):
    """Select the highest scoring ceil(ratio * size) members per cluster.

    Noise (label ``-1``) is excluded from both masks.  Equal scores are broken
    by original row index, making selection stable across devices.
    """
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("bridge ratio must be in (0, 1]")
    labels_tensor = torch.as_tensor(labels)
    scores_tensor = torch.as_tensor(scores)
    if labels_tensor.ndim != 1 or scores_tensor.ndim != 1:
        raise ValueError("bridge labels and scores must be one-dimensional")
    if labels_tensor.numel() != scores_tensor.numel():
        raise ValueError("bridge labels and scores have different lengths")
    if not bool(torch.isfinite(scores_tensor).all().item()):
        raise ValueError("bridge scores must all be finite")

    labels_np = labels_tensor.detach().cpu().numpy().astype(np.int64)
    scores_np = scores_tensor.detach().cpu().numpy().astype(np.float64)
    near_np = np.zeros(labels_np.shape[0], dtype=np.bool_)
    for cluster_id in sorted(set(labels_np.tolist())):
        if cluster_id == -1:
            continue
        indexes = np.flatnonzero(labels_np == cluster_id)
        count = max(1, int(math.ceil(float(ratio) * len(indexes))))
        order = np.lexsort((indexes, -scores_np[indexes]))
        near_np[indexes[order[:count]]] = True
    valid_np = labels_np != -1
    far_np = valid_np & ~near_np
    device = labels_tensor.device
    return (torch.as_tensor(near_np, dtype=torch.bool, device=device),
            torch.as_tensor(far_np, dtype=torch.bool, device=device))


@torch.no_grad()
def bridge_centers(features, labels, near_mask):
    """Build normalized means of normalized near members for each cluster."""
    if features.ndim != 2:
        raise ValueError("bridge features must be two-dimensional")
    labels_tensor = torch.as_tensor(labels, device=features.device).view(-1)
    near_tensor = torch.as_tensor(
        near_mask, device=features.device, dtype=torch.bool).view(-1)
    if features.size(0) != labels_tensor.numel() or (
            features.size(0) != near_tensor.numel()):
        raise ValueError("bridge features, labels and mask have different lengths")
    normalized = F.normalize(features, p=2, dim=1)
    cluster_ids = sorted({
        int(value) for value in labels_tensor.detach().cpu().tolist()
        if int(value) != -1
    })
    centers = []
    for cluster_id in cluster_ids:
        members = (labels_tensor == cluster_id) & near_tensor
        if not bool(members.any().item()):
            raise ValueError(
                "aerial cluster {} has no selected bridge member".format(
                    cluster_id))
        mean = normalized[members].mean(dim=0, keepdim=True)
        centers.append(F.normalize(mean, p=2, dim=1).squeeze(0))
    if not centers:
        return features.new_empty((0, features.size(1)))
    return torch.stack(centers, dim=0)


def bridge_selection_stats(labels, scores, near_mask):
    """Summarize valid clusters, near/far counts and their score means."""
    labels_tensor = torch.as_tensor(labels).view(-1)
    scores_tensor = torch.as_tensor(scores, dtype=torch.float64).view(-1)
    near_tensor = torch.as_tensor(near_mask, dtype=torch.bool).view(-1)
    if not (labels_tensor.numel() == scores_tensor.numel()
            == near_tensor.numel()):
        raise ValueError("bridge statistics inputs have different lengths")
    valid = labels_tensor != -1
    far = valid & ~near_tensor

    def _mean(mask):
        return (float(scores_tensor[mask].mean().item())
                if bool(mask.any().item()) else float("nan"))

    clusters = {int(value) for value in labels_tensor[valid].tolist()}
    return {
        "aerial_clusters": len(clusters),
        "near_images": int((valid & near_tensor).sum().item()),
        "far_images": int(far.sum().item()),
        "outliers": int((~valid).sum().item()),
        "near_score_mean": _mean(valid & near_tensor),
        "far_score_mean": _mean(far),
    }


def matching_difference_stats(bridge_aerial_to_ground,
                              full_aerial_to_ground, cluster_ids=None):
    """Return how often bridge and full centers choose different ground IDs."""
    if cluster_ids is None:
        cluster_ids = sorted(set(bridge_aerial_to_ground)
                             | set(full_aerial_to_ground))
    cluster_ids = [int(value) for value in cluster_ids]
    different = sum(
        bridge_aerial_to_ground.get(cluster_id)
        != full_aerial_to_ground.get(cluster_id)
        for cluster_id in cluster_ids)
    total = len(cluster_ids)
    return {
        "different_clusters": int(different),
        "total_clusters": int(total),
        "different_ratio": (float(different) / float(total)
                            if total else 0.0),
    }


def _cluster_majority_identities(labels, filenames):
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or labels.size != len(filenames):
        raise ValueError("GT labels and filenames have different lengths")
    identities = collections.defaultdict(list)
    for label, filename in zip(labels.tolist(), filenames):
        if int(label) != -1:
            identities[int(label)].append(parse_agreid_name(filename).pid)
    majority = {}
    for cluster_id, values in identities.items():
        counts = collections.Counter(values)
        majority[cluster_id] = sorted(
            counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return majority


def gt_match_accuracy(aerial_labels, aerial_filenames, ground_labels,
                      ground_filenames, aerial_to_ground):
    """Measure majority-identity agreement for matched aerial clusters."""
    aerial_majority = _cluster_majority_identities(
        aerial_labels, aerial_filenames)
    ground_majority = _cluster_majority_identities(
        ground_labels, ground_filenames)
    correct = 0
    evaluated = 0
    missing = 0
    for aerial_cluster, ground_cluster in sorted(aerial_to_ground.items()):
        aerial_cluster = int(aerial_cluster)
        ground_cluster = int(ground_cluster)
        if (aerial_cluster not in aerial_majority
                or ground_cluster not in ground_majority):
            missing += 1
            continue
        evaluated += 1
        correct += int(
            aerial_majority[aerial_cluster] == ground_majority[ground_cluster])
    return {
        "correct_pairs": int(correct),
        "evaluated_pairs": int(evaluated),
        "missing_pairs": int(missing),
        "accuracy": (float(correct) / float(evaluated)
                     if evaluated else float("nan")),
    }
