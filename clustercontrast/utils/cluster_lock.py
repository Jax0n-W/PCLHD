# -*- coding: utf-8 -*-
"""Optional process-wide lock for memory-heavy clustering sections."""

from __future__ import absolute_import, print_function

import contextlib
import os
import os.path as osp
import time

try:  # Linux training path required by the release.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on Windows.
    fcntl = None

try:  # Keep CPU unit tests usable on Windows development machines.
    import msvcrt
except ImportError:  # pragma: no cover - exercised only on POSIX.
    msvcrt = None


CLUSTER_LOCK_ENV = "PCLHD_CLUSTER_LOCK"


def resolve_cluster_lock_file(cli_value="", environ=None):
    """Use an explicit CLI path first, then ``PCLHD_CLUSTER_LOCK``."""
    environment = os.environ if environ is None else environ
    value = str(cli_value or environment.get(CLUSTER_LOCK_ENV, "")).strip()
    return osp.abspath(osp.expanduser(value)) if value else ""


def _lock(handle):
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return
    if msvcrt is None:
        raise RuntimeError("no supported file-lock implementation")
    handle.seek(0)
    while True:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            time.sleep(0.05)


def _unlock(handle):
    if fcntl is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def cluster_lock(lock_file="", label="clustering", printer=print):
    """Serialize a clustering block, or act as a zero-cost disabled context."""
    path = resolve_cluster_lock_file(lock_file)
    if not path:
        yield {"enabled": False, "wait_seconds": 0.0, "hold_seconds": 0.0}
        return

    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    handle = open(path, "a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        wait_started = time.monotonic()
        printer("[ClusterLock] waiting for {} ({})".format(path, label))
        _lock(handle)
        acquired = time.monotonic()
        stats = {
            "enabled": True,
            "wait_seconds": acquired - wait_started,
            "hold_seconds": 0.0,
            "path": path,
        }
        printer("[ClusterLock] acquired {} after {:.3f}s ({})".format(
            path, stats["wait_seconds"], label))
        try:
            yield stats
        finally:
            stats["hold_seconds"] = time.monotonic() - acquired
            _unlock(handle)
            printer("[ClusterLock] released {} after holding {:.3f}s ({})".format(
                path, stats["hold_seconds"], label))
    finally:
        handle.close()
