"""Camera-geometry-adaptive aspect jitter and path-aware loaders."""

from __future__ import absolute_import

import math
import os.path as osp
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from agreid_identity import parse_agreid_name
from . import IterLoader
from .agva_flag import (
    agva_flag_loader_enabled, get_dynamic_agva_process_with_flag,
    mask_geometry_ratio)
from .aspect_jitter import (
    _absolute_sample_path, _choose_samples, apply_horizontal_ratio,
    effective_aspect_max_ratio, save_aspect_preview)
from .dynamic_agva import get_dynamic_agva_process
from .sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
from .schp_mask_loader import SCHPMaskLoader


GEOMETRY_GAP_THRESHOLD = 0.02
MIN_MASK_COVERAGE = 0.80


def physical_camera(path):
    parsed = parse_agreid_name(path)
    if parsed.source == "official":
        return int(parsed.camid)
    return int(parsed.encoded_camid) % 10


def input_space_aspect(mask, image_size):
    """Return mask-box H/W divided by original crop H/W."""
    mask_ratio = mask_geometry_ratio(mask, image_size)
    if mask_ratio is None:
        return None
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    return float(mask_ratio) / (float(height) / float(width))


def _group_geometry(samples, image_root, mask_dir, group_function):
    loader = SCHPMaskLoader(
        mask_dir=mask_dir, image_root=image_root,
        resize_mismatch=True, strict=False)
    totals = defaultdict(int)
    values = defaultdict(list)
    samples_by_group = defaultdict(list)
    for sample in samples:
        path = _absolute_sample_path(sample, image_root)
        group = group_function(path)
        totals[group] += 1
        samples_by_group[group].append(sample)
        if loader._find(Path(path)) is None:
            continue
        with Image.open(path) as handle:
            size = handle.size
        mask = loader.load(path, img_shape=(size[1], size[0]))
        value = input_space_aspect(mask, size)
        if value is not None:
            values[group].append(value)
    return totals, values, samples_by_group


def compute_geometry_gaps(aerial_samples, ground_samples,
                          aerial_root, ground_root, aerial_mask_dir,
                          ground_mask_dir, max_ratio=0.3,
                          min_coverage=MIN_MASK_COVERAGE,
                          printer=print):
    """Measure per-camera input-space geometry without identity labels."""
    aerial_totals, aerial_values, aerial_groups = _group_geometry(
        aerial_samples, aerial_root, aerial_mask_dir,
        lambda unused: "aerial")
    ground_totals, ground_values, ground_groups = _group_geometry(
        ground_samples, ground_root, ground_mask_dir,
        lambda path: physical_camera(path))
    if not aerial_values.get("aerial"):
        raise RuntimeError("geometry aspect jitter found no valid aerial masks")
    aerial_median = float(np.median(aerial_values["aerial"]))
    stats = {}
    gaps = {}
    printer("[AspectGeometry] group   images   masks   coverage   median   gap")
    ordered = [("aerial", aerial_totals, aerial_values)]
    ordered.extend((camera, ground_totals, ground_values)
                   for camera in sorted(ground_totals))
    for group, totals, values in ordered:
        total = int(totals[group])
        valid = len(values.get(group, ()))
        coverage = valid / float(total) if total else 0.0
        if coverage < float(min_coverage):
            raise RuntimeError(
                "aspect geometry mask coverage below {:.0%} for {}: "
                "{}/{} ({:.2%})".format(
                    min_coverage, group, valid, total, coverage))
        median = float(np.median(values[group]))
        gap = 0.0 if group == "aerial" else float(np.clip(
            median / aerial_median - 1.0, 0.0, float(max_ratio)))
        stats[group] = {
            "images": total, "masks": valid, "coverage": coverage,
            "median": median, "gap": gap,
        }
        if group != "aerial":
            gaps[int(group)] = gap
        printer("[AspectGeometry] {:>6} {:8d} {:7d} {:10.2%} "
                "{:8.4f} {:7.4f}".format(
                    str(group), total, valid, coverage, median, gap))
    camera_counts = {
        int(camera): int(ground_totals[camera])
        for camera in ground_totals}
    return {
        "stats": stats,
        "gaps": gaps,
        "camera_counts": camera_counts,
        "aerial_groups": aerial_groups,
        "ground_groups": ground_groups,
    }


class GeometryAspectProcess(object):
    """Path-aware aspect jitter driven by measured physical-camera gaps."""

    def __init__(self, domain, camera_gaps, camera_counts, prob=0.5,
                 gap_threshold=GEOMETRY_GAP_THRESHOLD):
        if domain not in ("aerial", "ground"):
            raise ValueError("domain must be aerial or ground")
        self.domain = domain
        self.camera_gaps = {
            int(key): float(value) for key, value in camera_gaps.items()}
        self.camera_counts = {
            int(key): int(value) for key, value in camera_counts.items()}
        self.prob = float(prob)
        self.gap_threshold = float(gap_threshold)
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError("prob must be in [0, 1]")
        cameras = sorted(self.camera_counts)
        if not cameras or sum(self.camera_counts.values()) <= 0:
            raise ValueError("camera_counts must contain ground images")
        self.cameras = cameras
        self.weights = [self.camera_counts[camera] for camera in cameras]
        self.last_ratio = 1.0
        self.last_camera = None

    def sample_camera(self):
        return int(random.choices(
            self.cameras, weights=self.weights, k=1)[0])

    def sample_ratio(self, path=None):
        if self.domain == "ground":
            camera = physical_camera(path)
        else:
            camera = self.sample_camera()
        gap = float(self.camera_gaps.get(camera, 0.0))
        self.last_camera = camera
        if gap <= self.gap_threshold:
            return 1.0
        limit = math.log1p(gap)
        if self.domain == "ground":
            return math.exp(random.uniform(0.0, limit))
        return math.exp(random.uniform(-limit, 0.0))

    def __call__(self, image, path):
        self.last_ratio = 1.0
        self.last_camera = None
        if self.domain == "ground":
            camera = physical_camera(path)
            if self.camera_gaps.get(camera, 0.0) <= self.gap_threshold:
                self.last_camera = camera
                return image
        if self.prob <= 0.0 or random.random() >= self.prob:
            return image
        self.last_ratio = self.sample_ratio(path)
        if self.last_ratio == 1.0:
            return image
        return apply_horizontal_ratio(image, self.last_ratio)


def _geometry_preview_rows(args, result, aerial_samples, ground_samples,
                           aerial_root, ground_root):
    rows = []
    gaps = result["gaps"]
    max_gap = max(gaps.values()) if gaps else 0.0
    for index, sample in enumerate(_choose_samples(
            aerial_samples, 4, int(getattr(args, "seed", 1)))):
        path = _absolute_sample_path(sample, aerial_root)
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        rows.append(("aerial #{}".format(index + 1), image,
                     (1.0 / (1.0 + max_gap), 1.0, 1.0)))
    by_camera = defaultdict(list)
    for sample in ground_samples:
        path = _absolute_sample_path(sample, ground_root)
        by_camera[physical_camera(path)].append(sample)
    for camera in sorted(by_camera):
        gap = gaps.get(camera, 0.0)
        selected = _choose_samples(
            by_camera[camera], 4,
            int(getattr(args, "seed", 1)) + camera + 1)
        for index, sample in enumerate(selected):
            path = _absolute_sample_path(sample, ground_root)
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            rows.append(("ground C{} #{}".format(camera, index + 1), image,
                         (1.0, 1.0, 1.0 + gap)))
    return rows


def prepare_geometry_aspect(args, aerial_samples, ground_samples,
                            aerial_root, ground_root, preview_path):
    ground_mask_dir = getattr(args, "aspect_jitter_ground_mask_dir", "")
    if not ground_mask_dir:
        if getattr(args, "dataset_key", "agreid") == "agreidv2":
            ground_mask_dir = args.dynamic_agva_mask_dir
            args.aspect_jitter_ground_mask_dir = ground_mask_dir
        else:
            raise ValueError(
                "--aspect-jitter-ground-mask-dir is required for geometry "
                "mode on AG-ReID.v1/LAGPeR")
    configured_root = getattr(args, "dynamic_agva_image_root", "")
    effective_aerial_root = configured_root or aerial_root
    if getattr(args, "dataset_key", "agreid") == "agreidv2":
        effective_ground_root = configured_root or ground_root
    else:
        effective_ground_root = ground_root
    result = compute_geometry_gaps(
        aerial_samples, ground_samples,
        effective_aerial_root, effective_ground_root,
        args.dynamic_agva_mask_dir, ground_mask_dir,
        max_ratio=effective_aspect_max_ratio(args))
    args._aspect_geometry_result = result
    args._aspect_geometry_aerial_process = GeometryAspectProcess(
        "aerial", result["gaps"], result["camera_counts"],
        prob=args.aspect_jitter_prob)
    args._aspect_geometry_ground_process = GeometryAspectProcess(
        "ground", result["gaps"], result["camera_counts"],
        prob=args.aspect_jitter_prob)
    rows = _geometry_preview_rows(
        args, result, aerial_samples, ground_samples,
        effective_aerial_root, effective_ground_root)
    save_aspect_preview(rows, preview_path)
    return result


class _PreprocessorGeometry(Dataset):
    def __init__(self, dataset, root, transform, geometry_process,
                 image_process=None, resolution_process=None,
                 flagged=False):
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.geometry_process = geometry_process
        self.image_process = image_process
        self.resolution_process = resolution_process
        self.flagged = bool(flagged)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname, pid, camid = self.dataset[index]
        path = _absolute_sample_path((fname,), self.root)
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        is_aug = False
        if self.image_process is not None:
            result = self.image_process(image, path)
            if self.flagged:
                image, is_aug = result
            else:
                image = result
        image = self.geometry_process(image, path)
        if self.resolution_process is not None:
            image = self.resolution_process(image, path)
        if self.transform is not None:
            image = self.transform(image)
        base = (image, fname, pid, camid, index)
        return base + (bool(is_aug),) if self.flagged else base


class _PreprocessorColorGeometry(Dataset):
    def __init__(self, dataset, root, transform, transform1,
                 geometry_process, resolution_process1=None):
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.transform1 = transform1
        self.geometry_process = geometry_process
        self.resolution_process1 = resolution_process1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname, pid, camid = self.dataset[index]
        path = _absolute_sample_path((fname,), self.root)
        with Image.open(path) as handle:
            original = handle.convert("RGB")
        first = self.geometry_process(original.copy(), path)
        second = self.geometry_process(original.copy(), path)
        if self.resolution_process1 is not None:
            second = self.resolution_process1(second, path)
        return (self.transform(first), self.transform1(second), fname,
                pid, camid, index)


def _sampler(train_set, num_instances, no_cam):
    if num_instances <= 0:
        return None
    sampler_class = (RandomMultipleGallerySamplerNoCam if no_cam else
                     RandomMultipleGallerySampler)
    return sampler_class(train_set, num_instances)


def get_train_loader_ir_geometry(args, dataset, height, width, batch_size,
                                 workers, num_instances, iters,
                                 trainset=None, no_cam=False,
                                 train_transformer=None,
                                 resolution_process=None):
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    sampler = _sampler(train_set, num_instances, no_cam)
    flagged = agva_flag_loader_enabled(args)
    image_process = (get_dynamic_agva_process_with_flag(args, dataset.images_dir)
                     if flagged else
                     get_dynamic_agva_process(args, dataset.images_dir))
    preprocessor = _PreprocessorGeometry(
        train_set, dataset.images_dir, train_transformer,
        args._aspect_geometry_aerial_process,
        image_process=image_process, resolution_process=resolution_process,
        flagged=flagged)
    return IterLoader(DataLoader(
        preprocessor, batch_size=batch_size, num_workers=workers,
        sampler=sampler, shuffle=sampler is None, pin_memory=True,
        drop_last=True), length=iters)


def get_train_loader_color_geometry(args, dataset, height, width, batch_size,
                                    workers, num_instances, iters,
                                    trainset=None, no_cam=False,
                                    train_transformer=None,
                                    train_transformer1=None,
                                    resolution_process=None):
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    sampler = _sampler(train_set, num_instances, no_cam)
    preprocessor = _PreprocessorColorGeometry(
        train_set, dataset.images_dir, train_transformer,
        train_transformer1, args._aspect_geometry_ground_process,
        resolution_process1=resolution_process)
    return IterLoader(DataLoader(
        preprocessor, batch_size=batch_size, num_workers=workers,
        sampler=sampler, shuffle=sampler is None, pin_memory=True,
        drop_last=True), length=iters)
