from __future__ import absolute_import

import multiprocessing
import os
import tempfile
import time
import unittest

from clustercontrast.utils.cluster_lock import (
    cluster_lock, resolve_cluster_lock_file)
import experiment_runner as runner
from scripts import run_all_ablations as batch_runner


def _lock_worker(lock_path, gate, output, hold_seconds):
    gate.wait()
    with cluster_lock(lock_path, label="unit-test", printer=lambda value: None):
        entered = time.monotonic()
        time.sleep(hold_seconds)
        exited = time.monotonic()
    output.put((entered, exited))


class ClusterLockTests(unittest.TestCase):

    def test_disabled_lock_is_noop(self):
        with cluster_lock("", printer=lambda value: None) as stats:
            self.assertFalse(stats["enabled"])
        self.assertEqual(resolve_cluster_lock_file("", environ={}), "")

    def test_environment_fallback(self):
        path = resolve_cluster_lock_file(
            "", environ={"PCLHD_CLUSTER_LOCK": "example.lock"})
        self.assertTrue(path.endswith("example.lock"))

    def test_two_processes_hold_lock_serially(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            lock_path = os.path.join(directory, "cluster.lock")
            gate = context.Event()
            output = context.Queue()
            processes = [
                context.Process(
                    target=_lock_worker,
                    args=(lock_path, gate, output, 0.25))
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            gate.set()
            intervals = [output.get(timeout=10.0) for _ in processes]
            for process in processes:
                process.join(timeout=10.0)
                self.assertEqual(process.exitcode, 0)
            intervals.sort()
            self.assertGreaterEqual(
                intervals[1][0] + 0.02, intervals[0][1])

    def test_runner_forwards_explicit_lock_to_every_engine(self):
        for experiment in ("baseline", "three_domain", "bridge"):
            args = runner.build_parser().parse_args([
                "--experiment", experiment, "--cluster-lock-file",
                "shared.lock", "--dry-run"])
            stage1_name, stage2_name = runner.default_log_names(args)
            stage1 = runner.build_stage1_command(
                args, stage1_name, stage2_name)
            stage2 = runner.build_stage2_command(
                args, "/tmp/model_best.pth.tar",
                stage1_name, stage2_name)
            self.assertIn("--cluster-lock-file", stage1)
            self.assertIn("--cluster-lock-file", stage2)

    def test_batch_runner_accepts_bridge_and_shared_lock(self):
        args = batch_runner.build_parser().parse_args([
            "--experiments", "bridge", "dynamic_bridge",
            "--cluster-lock-file", "shared.lock", "--dry-run"])
        self.assertEqual(args.experiments, ["bridge", "dynamic_bridge"])
        self.assertIn("--cluster-lock-file", batch_runner.common_args(args))


if __name__ == "__main__":
    unittest.main()
