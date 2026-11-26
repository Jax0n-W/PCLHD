"""Single-link consolidation of aerial DBSCAN clusters and GT diagnostics."""

import math
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from agreid_identity import parse_agreid_name


AERIAL_MERGE_MODES = ("off", "threshold", "adaptive")
DEFAULT_TAU = 0.7
DEFAULT_LAMBDA = 0.9


def add_aerial_merge_arguments(parser):
    parser.add_argument("--aerial-merge", choices=AERIAL_MERGE_MODES,
                        default="off", help="Stage-2 aerial cluster consolidation")
    parser.add_argument("--aerial-merge-tau", type=float, default=DEFAULT_TAU,
                        help="cosine cutoff in threshold mode")
    parser.add_argument("--aerial-merge-lambda", type=float,
                        default=DEFAULT_LAMBDA,
                        help="compactness multiplier in adaptive mode")


def _validate_merge_options(mode, tau, lam):
    if mode not in AERIAL_MERGE_MODES:
        raise ValueError("unknown --aerial-merge mode: {}".format(mode))
    if mode == "threshold" and not -1.0 <= tau <= 1.0:
        raise ValueError("--aerial-merge-tau must be in [-1, 1]")
    if mode == "adaptive" and (not math.isfinite(lam) or lam < 0.0):
        raise ValueError("--aerial-merge-lambda must be finite and nonnegative")


def validate_aerial_merge_args(args, supports_bridge=True):
    if args.aerial_merge != "off" and not supports_bridge:
        raise ValueError(
            "--aerial-merge requires a bridge-engine experiment. "
            "Use --experiment bridge --bridge-ratio 1.0 for baseline + merge.")
    _validate_merge_options(
        args.aerial_merge, args.aerial_merge_tau, args.aerial_merge_lambda)


def aerial_merge_cli_args(args):
    if args.aerial_merge == "off":
        return []
    return [
        "--aerial-merge", args.aerial_merge,
        "--aerial-merge-tau", str(args.aerial_merge_tau),
        "--aerial-merge-lambda", str(args.aerial_merge_lambda),
    ]


def aerial_merge_log_suffix(args):
    if args.aerial_merge == "threshold":
        return "_am{:g}".format(args.aerial_merge_tau)
    if args.aerial_merge == "adaptive":
        return "_ama{:g}".format(args.aerial_merge_lambda)
    return ""


def _component_roots(edges, num_clusters):
    parent = list(range(num_clusters))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in edges:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            # Cluster indices are sorted by old label.
            parent[max(left_root, right_root)] = min(left_root, right_root)
    return [find(index) for index in range(num_clusters)]


@torch.no_grad()
def merge_aerial_clusters(features, labels, mode, tau=DEFAULT_TAU,
                          lam=DEFAULT_LAMBDA):
    """Return (new_labels, num_clusters, stats) using only features and labels."""
    _validate_merge_options(mode, tau, lam)
    old_labels = np.asarray(labels, dtype=np.int64)
    if old_labels.ndim != 1:
        raise ValueError("aerial labels must be one-dimensional")
    valid = old_labels != -1
    cluster_ids, member_clusters, counts = np.unique(
        old_labels[valid], return_inverse=True, return_counts=True)
    num_before = len(cluster_ids)
    stats = {
        "clusters_before": num_before,
        "clusters_after": num_before,
        "merges": 0,
        "merged_groups": 0,
        "largest_group": int(num_before > 0),
    }
    if mode == "off":
        return labels, num_before, stats
    if features.ndim != 2 or features.size(0) != len(old_labels):
        raise ValueError("aerial features must have one row per label")
    if mode == "adaptive":
        stats["median_compactness"] = float("nan")
    if num_before == 0:
        return old_labels.copy(), 0, stats

    # CPU reductions avoid device-dependent atomic accumulation order.
    members = features.cpu()[valid]
    if members.dtype in (torch.float16, torch.bfloat16):
        members = members.float()
    indices = torch.from_numpy(member_clusters)
    sizes = torch.from_numpy(counts).to(members.dtype)
    sums = members.new_zeros((num_before, members.size(1)))
    sums.index_add_(0, indices, members)
    centers = F.normalize(sums / sizes[:, None], dim=1)
    similarity = (centers @ centers.T).clamp(-1.0, 1.0)
    cutoff = tau
    if mode == "adaptive":
        normalized_sums = torch.zeros_like(sums)
        normalized_sums.index_add_(0, indices, F.normalize(members, dim=1))
        compactness = (normalized_sums * centers).sum(dim=1) / sizes
        stats["median_compactness"] = float(np.median(compactness.numpy()))
        cutoff = lam * (compactness[:, None] + compactness[None, :]) / 2

    edges = torch.triu(similarity >= cutoff, diagonal=1).nonzero().tolist()
    roots = _component_roots(edges, num_before)
    _, cluster_labels, group_sizes = np.unique(
        roots, return_inverse=True, return_counts=True)
    num_after = len(group_sizes)
    new_labels = old_labels.copy()
    new_labels[valid] = cluster_labels[member_clusters]
    stats.update(
        clusters_after=num_after,
        merges=num_before - num_after,
        merged_groups=int(np.count_nonzero(group_sizes >= 2)),
        largest_group=int(group_sizes.max()),
    )
    return new_labels, num_after, stats


def _bcubed(identities_by_cluster, identity_counts):
    """Compute sample-averaged BCubed on the clustered samples."""
    total = sum(identity_counts.values())
    if total == 0:
        return dict(precision=float("nan"), recall=float("nan"), f=float("nan"))
    precision = recall = 0.0
    for counts in identities_by_cluster.values():
        cluster_size = sum(counts.values())
        for pid, overlap in counts.items():
            precision += overlap ** 2 / cluster_size
            recall += overlap ** 2 / identity_counts[pid]
    precision /= total
    recall /= total
    return dict(precision=precision, recall=recall,
                f=2 * precision * recall / (precision + recall))


def _pair_count(count):
    return count * (count - 1) // 2


def merge_gt_stats(old_labels, new_labels, names):
    """Measure pair precision and BCubed before and after an aerial merge."""
    old_labels = np.asarray(old_labels, dtype=np.int64)
    new_labels = np.asarray(new_labels, dtype=np.int64)
    if old_labels.shape != (len(names),) or new_labels.shape != old_labels.shape:
        raise ValueError("aerial labels and filenames must have matching lengths")
    if not np.array_equal(old_labels == -1, new_labels == -1):
        raise ValueError("aerial merge must preserve outliers")

    before = defaultdict(Counter)
    after = defaultdict(Counter)
    identity_counts = Counter()
    old_to_new = {}
    for old, new, name in zip(old_labels, new_labels, names):
        if old == -1:
            continue
        if old_to_new.setdefault(old, new) != new:
            raise ValueError("aerial merge cannot split an old cluster")
        pid = parse_agreid_name(name).pid
        before[old][pid] += 1
        after[new][pid] += 1
        identity_counts[pid] += 1

    majorities_by_group = defaultdict(Counter)
    for old, counts in before.items():
        majority = min(counts, key=lambda pid: (-counts[pid], pid))
        majorities_by_group[old_to_new[old]][majority] += 1
    merged_pairs = sum(_pair_count(sum(counts.values()))
                       for counts in majorities_by_group.values())
    correct_pairs = sum(_pair_count(count)
                        for counts in majorities_by_group.values()
                        for count in counts.values())
    return {
        "pair_precision": (correct_pairs / merged_pairs
                           if merged_pairs else float("nan")),
        "merged_pairs": merged_pairs,
        "bcubed_before": _bcubed(before, identity_counts),
        "bcubed_after": _bcubed(after, identity_counts),
    }
