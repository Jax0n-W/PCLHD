# -*- coding: utf-8 -*-
"""Controlled AG-ReID baseline, three-domain, bridge, and confidence launcher.

Both AG-ReID.v1 and AG-ReID.v2 retain the original 2x2 D-AGVA/three-domain
matrix and add opt-in bridge and confidence-aware Stage-2 experiments.
Channel augmentation stays fixed off. v1 keeps its historical default paths;
v2 derives SCHP and view-score paths from ``--data-dir`` when not explicit.
"""
from __future__ import absolute_import, print_function

import argparse
import os
import os.path as osp
import shlex
import subprocess
import sys

from clustercontrast.utils.data.resolution_augment import (
    add_resolution_augment_arguments, append_resolution_log_suffix,
    resolution_log_suffix, validate_resolution_augment_args)
from clustercontrast.utils.data.aspect_jitter import (
    ASPECT_JITTER_MODES, ASPECT_JITTER_STAGES,
    aspect_jitter_enabled, effective_aspect_max_ratio,
    validate_aspect_jitter_args)


PROJECT_ROOT = osp.dirname(osp.abspath(__file__))
DEFAULT_EPS = 0.3

EXPERIMENTS = {
    "baseline": {"dynamic_agva": False, "three_domain": False},
    "dynamic": {"dynamic_agva": True, "three_domain": False},
    "three_domain": {"dynamic_agva": False, "three_domain": True},
    "full": {"dynamic_agva": True, "three_domain": True},
    "bridge": {
        "dynamic_agva": False, "three_domain": False, "bridge": True},
    "dynamic_bridge": {
        "dynamic_agva": True, "three_domain": False, "bridge": True},
    "confident": {
        "dynamic_agva": False, "three_domain": False, "bridge": True,
        "confident_cross": True},
    "dynamic_confident": {
        "dynamic_agva": True, "three_domain": False, "bridge": True,
        "confident_cross": True},
}

DEFAULT_DATA_DIR = "/home/lab338/Jaxon/dataset/AGreid_train"
DEFAULT_MASK_DIR = (
    "/home/lab338/Jaxon/dataset/AGreid_train_SCHP/"
    "aerial_modify/1/bounding_box_train"
)
DEFAULT_IMAGE_ROOT = (
    "/home/lab338/Jaxon/dataset/AGreid_train/"
    "aerial_modify/1/bounding_box_train"
)
DEFAULT_VIEW_SCORE = osp.join(
    PROJECT_ROOT, "view_scores", "agreid_aerial_view_scores_foreground.csv"
)
DEFAULT_V2_VIEW_SCORE = osp.join(
    PROJECT_ROOT, "view_domain", "outputs", "agreidv2",
    "agreidv2_aerial_view_scores_foreground.csv"
)


class _AGVAV2Store(argparse.Action):
    """Remember explicit AGVA-v2 options so non-dynamic runs can reject them."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, "_agva_v2_requested", True)


class _AGVAV2StoreTrue(argparse.Action):
    def __init__(self, option_strings, dest, default=False, required=False,
                 help=None):
        super(_AGVAV2StoreTrue, self).__init__(
            option_strings=option_strings, dest=dest, nargs=0,
            default=default, required=required, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        setattr(namespace, "_agva_v2_requested", True)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "AG-ReID.v1/v2 baseline, D-AGVA, three-domain, and bridge "
            "experiments; channel augmentation fixed off"
        )
    )
    parser.add_argument(
        "--experiment", choices=tuple(EXPERIMENTS), default="full")
    parser.add_argument(
        "--dataset-key", choices=("agreid", "agreidv2"),
        default="agreid",
        help="Dataset generation; agreid preserves all AG-ReID.v1 defaults")
    parser.add_argument(
        "--run-mode", choices=("full", "stage1", "stage2"), default="full")
    parser.add_argument(
        "--stage1-checkpoint", default="",
        help="Required in stage2 mode; model_best.pth.tar is the default policy")
    parser.add_argument(
        "--checkpoint-policy", choices=("best", "final", "legacy-best"),
        default="best",
        help=("Stage-2 initialization in full mode: best uses the Stage-1 "
              "model_best.pth.tar; final uses checkpoint.pth.tar. "
              "legacy-best is a deprecated alias of best."))

    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--logs-dir", default=osp.join(PROJECT_ROOT, "logs_agreid_dt"))
    parser.add_argument("--stage1-log-name", default="")
    parser.add_argument("--stage2-log-name", default="")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--num-instances", type=int, default=16)
    parser.add_argument("--k1", type=int, default=30)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument(
        "--eps-all", type=float, default=None,
        help="Stage-2 ALL DBSCAN eps; defaults to --eps")
    parser.add_argument(
        "--dbscan-backend", choices=("sklearn", "fast"),
        default="sklearn",
        help="DBSCAN implementation; sklearn preserves current behavior")
    parser.add_argument(
        "--dump-cluster-dist-epoch", type=int, default=-1,
        help="zero-based Stage-2 epoch whose ALL distance matrix is saved")
    parser.add_argument("--lr", type=float, default=0.00035)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--print-freq", type=int, default=50)

    parser.add_argument(
        "--dynamic-agva-mask-dir", default="",
        help=("SCHP mask root. Default: v1 historical path; v2 "
              "<data-dir>_SCHP/train_all"))
    parser.add_argument(
        "--dynamic-agva-image-root", default="",
        help=("Image root mirrored by the mask root. Default: v1 historical "
              "path; v2 <data-dir>/AG-ReID.v2/train_all"))
    parser.add_argument(
        "--dynamic-agva-slots", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--dynamic-agva-min-displacement", type=float, default=0.030)
    parser.add_argument(
        "--dynamic-agva-max-displacement", type=float, default=0.040)
    parser.add_argument("--dynamic-agva-strict-masks", action="store_true")
    parser.add_argument(
        "--dynamic-agva-no-memory-update", action=_AGVAV2StoreTrue,
        help="augmented aerial samples do not update cluster memories")
    parser.add_argument(
        "--dynamic-agva-target", choices=("fixed", "ground-mixture"),
        default="fixed", action=_AGVAV2Store)
    parser.add_argument(
        "--dynamic-agva-need-scale", type=float, default=0.20,
        action=_AGVAV2Store)
    parser.add_argument(
        "--dynamic-agva-ground-mask-dir", default="",
        action=_AGVAV2Store)
    parser.set_defaults(_agva_v2_requested=False)
    parser.add_argument(
        "--aspect-jitter", choices=ASPECT_JITTER_MODES, default="off")
    parser.add_argument(
        "--aspect-jitter-max-ratio", type=float, default=None,
        help="default: 0.2 for fixed modes, 0.3 for geometry")
    parser.add_argument(
        "--aspect-jitter-prob", type=float, default=0.5)
    parser.add_argument(
        "--aspect-jitter-stages", type=int,
        choices=ASPECT_JITTER_STAGES, default=1)
    parser.add_argument("--aspect-jitter-ground-mask-dir", default="")

    parser.add_argument(
        "--view-score-file", default="",
        help=("Aerial OOF view-score CSV. Default: v1 bundled CSV; v2 "
              "view_domain/outputs/agreidv2/"
              "agreidv2_aerial_view_scores_foreground.csv"))
    parser.add_argument(
        "--view-score-column", default="p_ground_foreground")
    parser.add_argument(
        "--view-score-path-mode",
        choices=("absolute", "basename", "relative"), default="basename")
    parser.add_argument("--view-near-ratio", type=float, default=0.50)
    parser.add_argument("--far-batch-size", type=int, default=32)
    parser.add_argument("--near-batch-size", type=int, default=32)
    parser.add_argument(
        "--aerial-loss-weight-mode", choices=("dataset", "balanced"),
        default="dataset")
    parser.add_argument("--chain-cross-weight", type=float, default=0.25)
    parser.add_argument("--cluster-n-jobs", type=int, default=1)
    parser.add_argument(
        "--bridge-score", choices=("ground_knn", "view_csv", "random"),
        default="ground_knn")
    parser.add_argument("--bridge-topk", type=int, default=10)
    parser.add_argument("--bridge-ratio", type=float, default=0.50)
    parser.add_argument(
        "--bgm-fallback", choices=("pclhd", "nearest"), default="pclhd")
    parser.add_argument("--cross-retain-start", type=float, default=0.4)
    parser.add_argument("--cross-retain-end", type=float, default=1.0)
    parser.add_argument(
        "--cross-retain-ramp-epochs", type=int, default=20)
    parser.add_argument(
        "--cross-confidence", choices=("margin", "similarity"),
        default="margin")
    parser.add_argument("--log-gt-match-accuracy", action="store_true")
    parser.add_argument(
        "--cluster-lock-file", default="",
        help=("shared clustering flock path; empty uses PCLHD_CLUSTER_LOCK "
              "or leaves locking disabled"))

    parser.add_argument(
        "--debug-nonfinite", action="store_true", default=True)
    parser.add_argument(
        "--no-debug-nonfinite", action="store_false",
        dest="debug_nonfinite")
    parser.add_argument("--allow-log-overwrite", action="store_true")
    add_resolution_augment_arguments(parser)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration and print commands without executing")
    return parser


def dataset_default_paths(dataset_key, data_dir):
    """Return the default D-AGVA and view-score paths for one dataset."""
    if dataset_key == "agreid":
        return {
            "mask_dir": DEFAULT_MASK_DIR,
            "image_root": DEFAULT_IMAGE_ROOT,
            "view_score": DEFAULT_VIEW_SCORE,
        }
    if dataset_key == "agreidv2":
        outer_root = _path(data_dir).rstrip(os.sep)
        return {
            "mask_dir": osp.join(outer_root + "_SCHP", "train_all"),
            "image_root": osp.join(outer_root, "AG-ReID.v2", "train_all"),
            "view_score": DEFAULT_V2_VIEW_SCORE,
        }
    raise ValueError("Unknown dataset key: {}".format(dataset_key))


def resolve_dataset_paths(args):
    """Fill empty path arguments with dataset defaults (idempotent)."""
    defaults = dataset_default_paths(args.dataset_key, args.data_dir)
    if not args.dynamic_agva_mask_dir:
        args.dynamic_agva_mask_dir = defaults["mask_dir"]
    if not args.dynamic_agva_image_root:
        args.dynamic_agva_image_root = defaults["image_root"]
    if not args.view_score_file:
        args.view_score_file = defaults["view_score"]
    if (args.dataset_key == "agreidv2"
            and args.dynamic_agva_target == "ground-mixture"
            and not args.dynamic_agva_ground_mask_dir):
        args.dynamic_agva_ground_mask_dir = args.dynamic_agva_mask_dir
    if (args.dataset_key == "agreidv2"
            and args.aspect_jitter == "geometry"
            and not args.aspect_jitter_ground_mask_dir):
        args.aspect_jitter_ground_mask_dir = args.dynamic_agva_mask_dir
    return args


def get_experiment_spec(name):
    try:
        return dict(EXPERIMENTS[name])
    except KeyError:
        raise ValueError("Unknown experiment: {}".format(name))


def eps_log_suffix(eps):
    """Keep legacy names at 0.3 and isolate every non-default eps run."""
    value = float(eps)
    if abs(value - DEFAULT_EPS) < 1e-12:
        return ""
    return "_eps{:g}".format(value)


def _append_eps_suffix(name, eps):
    suffix = eps_log_suffix(eps)
    if not suffix or str(name).endswith(suffix):
        return name
    return "{}{}".format(name, suffix)


def resolve_eps_all(args):
    value = getattr(args, "eps_all", None)
    return float(args.eps if value is None else value)


def eps_all_log_suffix(eps, eps_all):
    value = float(eps if eps_all is None else eps_all)
    if abs(value - float(eps)) < 1e-12:
        return ""
    return "_epsall{:g}".format(value)


def _append_eps_all_suffix(name, eps, eps_all):
    suffix = eps_all_log_suffix(eps, eps_all)
    if not suffix or str(name).endswith(suffix):
        return name
    return "{}{}".format(name, suffix)


def _append_stage2_eps_suffix(name, eps, eps_all):
    suffix = eps_log_suffix(eps) + eps_all_log_suffix(eps, eps_all)
    if not suffix or str(name).endswith(suffix):
        return name
    result = _append_eps_suffix(name, eps)
    return _append_eps_all_suffix(result, eps, eps_all)


def default_log_names(args, spec=None):
    spec = spec or get_experiment_spec(args.experiment)
    d_flag = int(spec["dynamic_agva"])
    t_flag = int(spec["three_domain"])
    prefix = "agreidv2_dt" if args.dataset_key == "agreidv2" else "agreid_dt"
    agva_tag = agva_v2_log_tag(args)
    aspect_tag = aspect_jitter_log_tag(args)
    stage1 = args.stage1_log_name or (
        "{}_s1_d{}{}{}_channel_off_seed{}".format(
            prefix, d_flag, agva_tag, aspect_tag, args.seed))
    fallback_suffix = "_nnfb" if args.bgm_fallback == "nearest" else ""
    if spec.get("confident_cross", False):
        stage2 = args.stage2_log_name or (
            "{}_{}_d{}{}{}_cr{}-{}r{}{}_channel_off_s2_seed{}".format(
                prefix, args.experiment, d_flag, agva_tag, aspect_tag,
                str(float(args.cross_retain_start)),
                str(float(args.cross_retain_end)),
                args.cross_retain_ramp_epochs, fallback_suffix, args.seed))
    elif spec.get("bridge", False):
        stage2 = args.stage2_log_name or (
            "{}_{}_d{}{}{}_b1{}_channel_off_s2_seed{}".format(
                prefix, args.experiment, d_flag, agva_tag, aspect_tag,
                fallback_suffix, args.seed))
    else:
        stage2 = args.stage2_log_name or (
            "{}_{}_d{}{}{}_t{}_channel_off_s2_seed{}".format(
                prefix, args.experiment, d_flag, agva_tag, aspect_tag, t_flag,
                args.seed))
    stage1 = append_resolution_log_suffix(stage1, args)
    stage2 = append_resolution_log_suffix(stage2, args)
    stage1 = _append_eps_suffix(stage1, args.eps)
    stage2 = _append_stage2_eps_suffix(
        stage2, args.eps, resolve_eps_all(args))
    return stage1, stage2


def _path(value):
    return osp.abspath(osp.expanduser(value))


def _base_common_args(args, stage1_log_name, stage2_log_name):
    dataset_name = (
        "agreidv2_ir" if args.dataset_key == "agreidv2" else "agreid_ir")
    result = [
        "--dataset", dataset_name,
        "--trial", str(args.trial),
        "--epochs", str(args.epochs),
        "--iters", str(args.iters),
        "--step-size", str(args.step_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--batch-size", str(args.batch_size),
        "--height", str(args.height),
        "--width", str(args.width),
        "--num-instances", str(args.num_instances),
        "--eps", str(args.eps),
        "--k1", str(args.k1),
        "--k2", str(args.k2),
        "--temp", str(args.temp),
        "--momentum", str(args.momentum),
        "--seed", str(args.seed),
        "--print-freq", str(args.print_freq),
        "--pooling-type", "gem",
        "--data-dir", _path(args.data_dir),
        "--logs-dir", _path(args.logs_dir),
        "--stage1-log-name", stage1_log_name,
        "--stage2-log-name", stage2_log_name,
        "--workers", str(args.workers),
        "--channel-adap-gray", "off",
    ]
    if args.debug_nonfinite:
        result.append("--debug-nonfinite")
    if args.allow_log_overwrite:
        result.append("--allow-log-overwrite")
    if args.dataset_key != "agreid":
        result += ["--dataset-key", args.dataset_key]
    if args.dbscan_backend != "sklearn":
        result += ["--dbscan-backend", args.dbscan_backend]
    if args.cluster_lock_file:
        result += ["--cluster-lock-file", _path(args.cluster_lock_file)]
    return result


def _dynamic_args(args):
    result = [
        "--use-dynamic-agva",
        "--dynamic-agva-mask-dir", _path(args.dynamic_agva_mask_dir),
        "--dynamic-agva-image-root", _path(args.dynamic_agva_image_root),
        "--dynamic-agva-slots", str(args.dynamic_agva_slots),
        "--dynamic-agva-min-displacement",
        str(args.dynamic_agva_min_displacement),
        "--dynamic-agva-max-displacement",
        str(args.dynamic_agva_max_displacement),
    ]
    if args.dynamic_agva_strict_masks:
        result.append("--dynamic-agva-strict-masks")
    return result


def _agva_v2_args(args):
    """Return only the opt-in AGVA-v2 arguments (merge-friendly hook)."""
    result = []
    if args.dynamic_agva_no_memory_update:
        result.append("--dynamic-agva-no-memory-update")
    if args.dynamic_agva_target == "ground-mixture":
        result += [
            "--dynamic-agva-target", "ground-mixture",
            "--dynamic-agva-need-scale", str(args.dynamic_agva_need_scale),
            "--dynamic-agva-ground-mask-dir",
            _path(args.dynamic_agva_ground_mask_dir),
        ]
    return result


def agva_v2_log_tag(args):
    """Return compact tags placed immediately after the dynamic flag."""
    tag = ""
    if getattr(args, "dynamic_agva_no_memory_update", False):
        tag += "mm"
    if getattr(args, "dynamic_agva_target", "fixed") == "ground-mixture":
        tag += "gm{:g}".format(float(args.dynamic_agva_need_scale))
    return tag


def _aspect_jitter_args(args, stage):
    """Return stage-scoped aspect-jitter arguments for every engine."""
    if not aspect_jitter_enabled(args, stage):
        return []
    result = [
        "--aspect-jitter", args.aspect_jitter,
        "--aspect-jitter-max-ratio", str(effective_aspect_max_ratio(args)),
        "--aspect-jitter-prob", str(args.aspect_jitter_prob),
        "--aspect-jitter-stages", str(args.aspect_jitter_stages),
    ]
    if args.aspect_jitter == "geometry":
        result += [
            "--aspect-jitter-ground-mask-dir",
            _path(args.aspect_jitter_ground_mask_dir),
            "--dynamic-agva-mask-dir", _path(args.dynamic_agva_mask_dir),
            "--dynamic-agva-image-root", _path(args.dynamic_agva_image_root),
        ]
    return result


def aspect_jitter_log_tag(args):
    mode = getattr(args, "aspect_jitter", "off")
    if mode == "off":
        return ""
    if mode == "geometry":
        tag = "ajgeop{:g}".format(float(args.aspect_jitter_prob))
    else:
        tag = "aj{}{:g}p{:g}".format(
            mode, effective_aspect_max_ratio(args),
            float(args.aspect_jitter_prob))
    if int(args.aspect_jitter_stages) != 1:
        tag += "s{}".format(args.aspect_jitter_stages)
    return tag


def _resolution_args(args):
    if args.res_aug_mode == "off":
        return []
    return [
        "--res-aug-mode", args.res_aug_mode,
        "--res-aug-prob", str(args.res_aug_prob),
        "--res-aug-domains", args.res_aug_domains,
        "--res-aug-stages", str(args.res_aug_stages),
        "--res-aug-min-scale", str(args.res_aug_min_scale),
        "--res-aug-patches",
        str(args.res_aug_patches[0]), str(args.res_aug_patches[1]),
        "--res-aug-patch-area",
        str(args.res_aug_patch_area[0]), str(args.res_aug_patch_area[1]),
    ]


def build_stage1_command(args, stage1_log_name=None, stage2_log_name=None):
    resolve_dataset_paths(args)
    spec = get_experiment_spec(args.experiment)
    default_s1, default_s2 = default_log_names(args, spec)
    stage1_log_name = stage1_log_name or default_s1
    stage2_log_name = stage2_log_name or default_s2
    command = [sys.executable, osp.join(PROJECT_ROOT, "_train_agreid_engine.py")]
    command += _base_common_args(args, stage1_log_name, stage2_log_name)
    command += ["--stage1-only"]
    command += _resolution_args(args)
    if spec["dynamic_agva"]:
        command += _dynamic_args(args)
        command += _agva_v2_args(args)
    command += _aspect_jitter_args(args, 1)
    return command


def build_stage2_command(args, checkpoint, stage1_log_name=None,
                         stage2_log_name=None):
    resolve_dataset_paths(args)
    spec = get_experiment_spec(args.experiment)
    default_s1, default_s2 = default_log_names(args, spec)
    stage1_log_name = stage1_log_name or default_s1
    stage2_log_name = stage2_log_name or default_s2
    checkpoint = _path(checkpoint)
    if spec.get("bridge", False):
        confident_cross = spec.get("confident_cross", False)
        command = [
            sys.executable,
            osp.join(PROJECT_ROOT, "_train_agreid_bridge_engine.py")]
        command += _base_common_args(args, stage1_log_name, stage2_log_name)
        command += [
            "--stage2-only", "--stage1-checkpoint", checkpoint,
            "--bridge-score", args.bridge_score,
            "--bridge-topk", str(args.bridge_topk),
            "--bridge-ratio", ("1.0" if confident_cross
                               else str(args.bridge_ratio)),
            "--bgm-fallback", args.bgm_fallback,
            "--view-score-file", _path(args.view_score_file),
            "--view-score-path-mode", args.view_score_path_mode,
        ]
        if confident_cross:
            command += [
                "--confident-cross",
                "--cross-retain-start", str(args.cross_retain_start),
                "--cross-retain-end", str(args.cross_retain_end),
                "--cross-retain-ramp-epochs",
                str(args.cross_retain_ramp_epochs),
                "--cross-confidence", args.cross_confidence,
            ]
        if args.log_gt_match_accuracy:
            command.append("--log-gt-match-accuracy")
        command += _resolution_args(args)
        command += _aspect_jitter_args(args, 2)
        if args.eps_all is not None:
            command += ["--eps-all", str(args.eps_all)]
        if args.dump_cluster_dist_epoch >= 0:
            command += [
                "--dump-cluster-dist-epoch",
                str(args.dump_cluster_dist_epoch)]
        return command

    if not spec["three_domain"]:
        command = [
            sys.executable, osp.join(PROJECT_ROOT, "_train_agreid_engine.py")]
        command += _base_common_args(args, stage1_log_name, stage2_log_name)
        command += [
            "--stage2-only", "--stage1-checkpoint", checkpoint,
            "--stage2-memorybank", "CMhard", "--no-ema-loss",
        ]
        command += _resolution_args(args)
        command += _aspect_jitter_args(args, 2)
        if args.eps_all is not None:
            command += ["--eps-all", str(args.eps_all)]
        if args.dump_cluster_dist_epoch >= 0:
            command += [
                "--dump-cluster-dist-epoch",
                str(args.dump_cluster_dist_epoch)]
        return command

    command = [
        sys.executable,
        osp.join(PROJECT_ROOT, "_train_agreid_three_domain_engine.py"),
        "--stage2-only",
        "--stage1-checkpoint", checkpoint,
        "--stage2-log-name", stage2_log_name,
        "--dataset",
        "agreidv2_ir" if args.dataset_key == "agreidv2" else "agreid_ir",
        "--trial", str(args.trial),
        "--epochs", str(args.epochs),
        "--iters", str(args.iters),
        "--step-size", str(args.step_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--batch-size", str(args.batch_size),
        "--test-batch", str(args.test_batch),
        "--height", str(args.height),
        "--width", str(args.width),
        "--num-instances", str(args.num_instances),
        "--eps", str(args.eps),
        "--k1", str(args.k1),
        "--k2", str(args.k2),
        "--temp", str(args.temp),
        "--momentum", str(args.momentum),
        "--seed", str(args.seed),
        "--print-freq", str(args.print_freq),
        "--eval-step", "1",
        "--pooling-type", "gem",
        "--data-dir", _path(args.data_dir),
        "--logs-dir", _path(args.logs_dir),
        "--workers", str(args.workers),
        "--view-score-file", _path(args.view_score_file),
        "--view-score-column", args.view_score_column,
        "--view-score-path-mode", args.view_score_path_mode,
        "--view-near-ratio", str(args.view_near_ratio),
        "--far-batch-size", str(args.far_batch_size),
        "--near-batch-size", str(args.near_batch_size),
        "--aerial-loss-weight-mode", args.aerial_loss_weight_mode,
        "--chain-cross-weight", str(args.chain_cross_weight),
        "--cluster-n-jobs", str(args.cluster_n_jobs),
        "--channel-adap-gray", "off",
    ]
    if args.dataset_key != "agreid":
        command += ["--dataset-key", args.dataset_key]
    if args.allow_log_overwrite:
        command.append("--allow-log-overwrite")
    if args.cluster_lock_file:
        command += ["--cluster-lock-file", _path(args.cluster_lock_file)]
    if args.eps_all is not None:
        command += ["--eps-all", str(args.eps_all)]
    if args.dbscan_backend != "sklearn":
        command += ["--dbscan-backend", args.dbscan_backend]
    if args.dump_cluster_dist_epoch >= 0:
        command += [
            "--dump-cluster-dist-epoch", str(args.dump_cluster_dist_epoch)]
    command += _resolution_args(args)
    command += _aspect_jitter_args(args, 2)
    return command


def checkpoint_path(args, stage1_log_name):
    filename = ("checkpoint.pth.tar"
                if args.checkpoint_policy == "final"
                else "model_best.pth.tar")
    return osp.join(
        _path(args.logs_dir), stage1_log_name, str(args.trial), filename)


def validate_args(args):
    resolve_dataset_paths(args)
    spec = get_experiment_spec(args.experiment)
    validate_resolution_augment_args(args)
    validate_aspect_jitter_args(args)
    if args.run_mode == "stage2" and not args.stage1_checkpoint:
        raise ValueError("--run-mode stage2 requires --stage1-checkpoint")
    if args.run_mode != "stage2" and args.stage1_checkpoint:
        raise ValueError(
            "--stage1-checkpoint is only valid with --run-mode stage2")
    if args.far_batch_size + args.near_batch_size != args.batch_size:
        raise ValueError(
            "far-batch-size + near-batch-size must equal batch-size")
    if not 0.0 < args.view_near_ratio < 1.0:
        raise ValueError("--view-near-ratio must be in (0, 1)")
    if not (0.0 < args.dynamic_agva_min_displacement <=
            args.dynamic_agva_max_displacement < 0.20):
        raise ValueError("invalid D-AGVA displacement range")
    if args.dynamic_agva_need_scale <= 0:
        raise ValueError("--dynamic-agva-need-scale must be positive")
    if getattr(args, "_agva_v2_requested", False) and not spec["dynamic_agva"]:
        raise ValueError(
            "D-AGVA memory/target options require a dynamic experiment")
    agva_v2_active = bool(_agva_v2_args(args))
    if agva_v2_active and args.experiment != "dynamic":
        raise ValueError(
            "D-AGVA memory masking/ground-mixture currently supports only "
            "the dynamic baseline trainer; bridge, confident, and "
            "three-domain trainers are unsupported")
    if (args.dynamic_agva_target == "ground-mixture"
            and args.dataset_key != "agreidv2"
            and not args.dynamic_agva_ground_mask_dir):
        raise ValueError(
            "--dynamic-agva-ground-mask-dir is required for "
            "ground-mixture on AG-ReID.v1/LAGPeR")
    if args.cluster_n_jobs < 1:
        raise ValueError("--cluster-n-jobs must be at least 1")
    if args.bridge_topk < 1:
        raise ValueError("--bridge-topk must be at least 1")
    if not 0.0 < args.bridge_ratio <= 1.0:
        raise ValueError("--bridge-ratio must be in (0, 1]")
    if not 0.0 < args.cross_retain_start <= args.cross_retain_end <= 1.0:
        raise ValueError(
            "cross retention must satisfy 0 < start <= end <= 1")
    if args.cross_retain_ramp_epochs < 1:
        raise ValueError("--cross-retain-ramp-epochs must be at least 1")
    if (args.dump_cluster_dist_epoch < -1
            or args.dump_cluster_dist_epoch >= args.epochs):
        raise ValueError(
            "--dump-cluster-dist-epoch must be -1 or a zero-based epoch "
            "smaller than --epochs")
    if (args.aspect_jitter == "geometry"
            and args.dataset_key != "agreidv2"
            and not args.aspect_jitter_ground_mask_dir):
        active_stage = (
            (args.run_mode in ("full", "stage1")
             and aspect_jitter_enabled(args, 1))
            or (args.run_mode in ("full", "stage2")
                and aspect_jitter_enabled(args, 2)))
        if active_stage:
            raise ValueError(
                "--aspect-jitter-ground-mask-dir is required for geometry "
                "mode on AG-ReID.v1/LAGPeR")

    if args.dry_run:
        return
    if not osp.isdir(_path(args.data_dir)):
        raise FileNotFoundError("dataset root not found: {}".format(
            args.data_dir))
    runs_stage1 = args.run_mode in ("full", "stage1")
    if runs_stage1 and spec["dynamic_agva"]:
        if not osp.isdir(_path(args.dynamic_agva_mask_dir)):
            raise FileNotFoundError("D-AGVA mask directory not found: {}".format(
                args.dynamic_agva_mask_dir))
        if not osp.isdir(_path(args.dynamic_agva_image_root)):
            raise FileNotFoundError("D-AGVA image root not found: {}".format(
                args.dynamic_agva_image_root))
    if args.run_mode == "stage2" and not osp.isfile(
            _path(args.stage1_checkpoint)):
        raise FileNotFoundError("Stage 1 checkpoint not found: {}".format(
            args.stage1_checkpoint))
    runs_stage2 = args.run_mode in ("full", "stage2")
    if args.aspect_jitter == "geometry":
        active_geometry = (
            (runs_stage1 and aspect_jitter_enabled(args, 1))
            or (runs_stage2 and aspect_jitter_enabled(args, 2)))
        if active_geometry:
            for label, path in (
                    ("aerial", args.dynamic_agva_mask_dir),
                    ("ground", args.aspect_jitter_ground_mask_dir)):
                if not osp.isdir(_path(path)):
                    raise FileNotFoundError(
                        "Aspect geometry {} mask directory not found: {}"
                        .format(label, path))
    needs_view_scores = (spec["three_domain"] or (
        spec.get("bridge", False) and args.bridge_score == "view_csv"))
    if runs_stage2 and needs_view_scores and not osp.isfile(
            _path(args.view_score_file)):
        raise FileNotFoundError("view-score file not found: {}".format(
            args.view_score_file))


def _run(command, dry_run):
    print("\n$ {}".format(shlex.join(command)))
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _print_config(args, spec, stage1_log_name, stage2_log_name):
    if args.dataset_key == "agreid":
        print("========== AG-ReID.v1 Controlled Ablation ==========")
    else:
        print("========== AG-ReID.v2 Controlled Ablation ==========")
    print("Experiment: {}".format(args.experiment))
    print("Run mode: {}".format(args.run_mode))
    print("D-AGVA (Stage 1 only): {}".format(spec["dynamic_agva"]))
    print("Three-domain matching (Stage 2): {}".format(
        spec["three_domain"]))
    print("Bridge matching (Stage 2): {}".format(
        spec.get("bridge", False)))
    print("ChannelAdapGray: False (Stage 1 and Stage 2)")
    print("ChannelExchange/Gray: False (Stage 1 and Stage 2)")
    print("ChannelRandomErasing: True (spatial augmentation)")
    print("Stage 2 memory: CMhard")
    print("Stage 2 EMA loss included: False")
    print("Stage 2 EMA encoder update enabled: True")
    print("Stage 2 ALL-memory optimization enabled: True")
    if args.res_aug_mode != "off":
        print("Resolution augmentation: mode={} p={} domains={} stages={} "
              "min_scale={} patches={}..{} patch_area={}..{}".format(
                  args.res_aug_mode, args.res_aug_prob,
                  args.res_aug_domains, args.res_aug_stages,
                  args.res_aug_min_scale, args.res_aug_patches[0],
                  args.res_aug_patches[1], args.res_aug_patch_area[0],
                  args.res_aug_patch_area[1]))
        print("Resolution log suffix: {}".format(
            resolution_log_suffix(args)))
    if args.aspect_jitter != "off":
        print("Aspect jitter: mode={} max_ratio={} prob={} stages={}"
              .format(args.aspect_jitter, effective_aspect_max_ratio(args),
                      args.aspect_jitter_prob, args.aspect_jitter_stages))
        print("Aspect jitter log tag: {}".format(
            aspect_jitter_log_tag(args)))
        if args.aspect_jitter == "geometry":
            print("Aspect jitter ground mask dir: {}".format(
                _path(args.aspect_jitter_ground_mask_dir)))
    if spec["dynamic_agva"]:
        print("D-AGVA mask dir: {}".format(_path(args.dynamic_agva_mask_dir)))
        print("D-AGVA image root: {}".format(
            _path(args.dynamic_agva_image_root)))
        if _agva_v2_args(args):
            print("D-AGVA no memory update: {}".format(
                args.dynamic_agva_no_memory_update))
            print("D-AGVA target: {}".format(args.dynamic_agva_target))
            print("D-AGVA need scale: {}".format(
                args.dynamic_agva_need_scale))
            if args.dynamic_agva_target == "ground-mixture":
                print("D-AGVA ground mask dir: {}".format(
                    _path(args.dynamic_agva_ground_mask_dir)))
    if spec["three_domain"]:
        print("View score file: {}".format(_path(args.view_score_file)))
        print("View near ratio (per camera): {}".format(args.view_near_ratio))
    if spec.get("bridge", False):
        effective_bridge_ratio = (
            1.0 if spec.get("confident_cross", False)
            else args.bridge_ratio)
        print("Bridge score: {}".format(args.bridge_score))
        print("Bridge top-k: {}".format(args.bridge_topk))
        print("Bridge ratio (within each cluster): {}".format(
            effective_bridge_ratio))
        print("BGM fallback: {}".format(args.bgm_fallback))
        print("Bridge GT match diagnostics: {}".format(
            args.log_gt_match_accuracy))
        if args.bridge_score == "view_csv":
            print("Bridge view score file: {}".format(
                _path(args.view_score_file)))
    if spec.get("confident_cross", False):
        print("Confidence-aware cross matching (Stage 2): True")
        print("Fixed bridge ratio: 1.0")
        print("Cross retention: start={}, end={}, ramp_epochs={}".format(
            args.cross_retain_start, args.cross_retain_end,
            args.cross_retain_ramp_epochs))
        print("Cross confidence: {}".format(args.cross_confidence))
    print("Cluster lock: {}".format(
        _path(args.cluster_lock_file) if args.cluster_lock_file else
        os.environ.get("PCLHD_CLUSTER_LOCK", "disabled")))
    print("Checkpoint policy: {}".format(args.checkpoint_policy))
    print("Stage 1 log: {}".format(stage1_log_name))
    print("Stage 2 log: {}".format(stage2_log_name))
    print("=====================================================")


def run_experiment(dataset_key="agreid", argv=None):
    parser = build_parser()
    parser.set_defaults(dataset_key=dataset_key)
    args = parser.parse_args(argv)
    validate_args(args)
    spec = get_experiment_spec(args.experiment)
    stage1_log_name, stage2_log_name = default_log_names(args, spec)
    _print_config(args, spec, stage1_log_name, stage2_log_name)

    checkpoint = ""
    if args.run_mode in ("full", "stage1"):
        _run(build_stage1_command(
            args, stage1_log_name, stage2_log_name), args.dry_run)
        checkpoint = checkpoint_path(args, stage1_log_name)
        if not args.dry_run and not osp.isfile(checkpoint):
            raise FileNotFoundError(
                "Stage 1 completed but checkpoint was not found: {}".format(
                    checkpoint))
    if args.run_mode == "stage1":
        return 0
    if args.run_mode == "stage2":
        checkpoint = _path(args.stage1_checkpoint)

    _run(build_stage2_command(
        args, checkpoint, stage1_log_name, stage2_log_name), args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_experiment("agreid"))
