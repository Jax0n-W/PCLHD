from __future__ import absolute_import, print_function

"""Query-height bucket helpers that never alter extracted features."""

import numpy as np

from .data.resolution_augment import read_original_height


def validate_resolution_buckets(thresholds):
    if thresholds is None:
        return None
    values = tuple(int(value) for value in thresholds)
    if not values:
        raise ValueError("--res-buckets requires at least one threshold")
    if any(value <= 0 for value in values):
        raise ValueError("--res-buckets thresholds must be positive")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(
            "--res-buckets thresholds must be strictly increasing")
    return values


def read_query_heights(paths):
    return np.asarray(
        [read_original_height(path) for path in paths], dtype=np.int64)


def resolution_bucket_masks(heights, thresholds):
    heights = np.asarray(heights, dtype=np.int64)
    thresholds = validate_resolution_buckets(thresholds)
    output = []
    lower = None
    for upper in thresholds:
        if lower is None:
            label = "<{}".format(upper)
            mask = heights < upper
        else:
            label = "{}-{}".format(lower, upper)
            mask = (heights >= lower) & (heights < upper)
        output.append((label, lower, upper, mask))
        lower = upper
    output.append((">={}".format(lower), lower, None, heights >= lower))
    return output


def evaluate_resolution_buckets(
        distmat, query_pids, gallery_pids, query_camids, gallery_camids,
        query_heights, thresholds, evaluator):
    """Evaluate query subsets against the unchanged full gallery matrix."""
    distmat = np.asarray(distmat)
    query_pids = np.asarray(query_pids)
    gallery_pids = np.asarray(gallery_pids)
    query_camids = np.asarray(query_camids)
    gallery_camids = np.asarray(gallery_camids)
    query_heights = np.asarray(query_heights)
    if distmat.shape[0] != len(query_heights):
        raise ValueError("query heights and distance rows must have equal length")

    rows = []
    for label, lower, upper, mask in resolution_bucket_masks(
            query_heights, thresholds):
        count = int(np.count_nonzero(mask))
        row = {
            "bucket": label,
            "query_height_min": "" if lower is None else int(lower),
            "query_height_max": "" if upper is None else int(upper),
            "query_images": count,
            "rank1": "",
            "rank5": "",
            "rank10": "",
            "mAP": "",
            "mINP": "",
        }
        if count:
            cmc, mean_ap, minp = evaluator(
                distmat[mask], query_pids[mask], gallery_pids,
                q_camids=query_camids[mask], g_camids=gallery_camids)
            row.update({
                "rank1": float(cmc[0]),
                "rank5": float(cmc[4]) if len(cmc) > 4 else "",
                "rank10": float(cmc[9]) if len(cmc) > 9 else "",
                "mAP": float(mean_ap),
                "mINP": float(minp),
            })
        rows.append(row)
    return rows


def format_bucket_result(protocol, row):
    if row["query_images"] == 0:
        metrics = "Rank-1=N/A mAP=N/A"
    else:
        metrics = "Rank-1={:.2%} mAP={:.2%}".format(
            row["rank1"], row["mAP"])
    return ("[RES-BUCKET] protocol={} bucket={} queries={} {}"
            .format(protocol, row["bucket"], row["query_images"], metrics))
