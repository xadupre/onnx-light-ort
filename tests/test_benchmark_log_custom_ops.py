"""Checks the gallery's custom-op paths and ORT-baseline speedups."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy
from onnx_light.onnx import ModelProto

from docs.examples.plot_benchmark_log_custom_ops import (
    LABELS,
    SIZES,
    make_model,
    measure,
    plot_results,
    summarize,
)
from docs.examples import plot_benchmark_log_custom_ops

LIBRARY = None


class TestBenchmarkLogCustomOps(unittest.TestCase):
    def test_downloaded_gallery_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "plot_benchmark_log_custom_ops.py"
            shutil.copyfile(plot_benchmark_log_custom_ops.__file__, script)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--library",
                    str(LIBRARY),
                    "--host",
                    str(LIBRARY),
                    "--backend",
                    "ort",
                    "--nodes",
                    "10",
                    "--sizes",
                    "7",
                    "--warmup",
                    "1",
                    "--repeat",
                    "2",
                    "--number",
                    "2",
                ],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
                cwd=directory,
            )
            result = json.loads(completed.stdout)
            self.assertEqual(result["backend"], "ort")
            self.assertEqual(result["api"], "iobinding")
            self.assertEqual(result["nodes"], 10)
            self.assertEqual(result["rows"][0]["size"], 7)

    def test_graph_topologies(self):
        for custom in (False, True):
            for nodes in (1, 10):
                with self.subTest(custom=custom, nodes=nodes):
                    model = ModelProto()
                    model.ParseFromString(make_model(custom, nodes))
                    self.assertEqual(len(model.graph.node), nodes)
                    name = "X"
                    for node in model.graph.node:
                        self.assertEqual(node.op_type, "Log")
                        self.assertEqual(node.domain, "onnx_light_cpu" if custom else "")
                        self.assertEqual(list(node.input), [name])
                        name = node.output[0]
                    self.assertEqual(name, "Y")

    def test_downloaded_native_worker_without_bridge_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "plot_benchmark_log_custom_ops.py"
            shutil.copyfile(plot_benchmark_log_custom_ops.__file__, script)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--backend",
                    "native",
                    "--nodes",
                    "10",
                    "--sizes",
                    "7",
                    "--warmup",
                    "1",
                    "--repeat",
                    "2",
                    "--number",
                    "2",
                ],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
                cwd=directory,
            )
            result = json.loads(completed.stdout)
            self.assertEqual(result["backend"], "native")
            self.assertEqual(result["api"], "run")
            self.assertIsNone(result["ort_version"])
            self.assertEqual(result["nodes"], 10)
            self.assertEqual(result["rows"][0]["size"], 7)

    def test_input_sizes_and_paths(self):
        self.assertGreater(len(SIZES), 5)
        self.assertEqual(SIZES, sorted(set(SIZES)))
        self.assertEqual(SIZES[0], 1)
        self.assertGreaterEqual(SIZES[-1], 1048576)
        self.assertEqual(set(LABELS), {"ort", "custom", "standalone", "native"})
        self.assertIn("without ONNX Runtime", LABELS["native"])
        self.assertIn("called by ONNX Runtime", LABELS["custom"])
        self.assertIn("ORT-ABI", LABELS["standalone"])
        self.assertIn("called by onnx-light", LABELS["standalone"])

    def test_speedups_and_plots(self):
        results = []
        for nodes in (1, 10):
            for backend, factor in (
                ("ort", 1),
                ("custom", 0.5),
                ("standalone", 2),
                ("native", 0.25),
            ):
                if backend == "standalone" and nodes != 1:
                    continue
                for trial in (1, 3):
                    results.append(
                        {
                            "backend": backend,
                            "nodes": nodes,
                            "rows": [
                                {"size": size, "samples_s": [trial * factor * nodes * size]}
                                for size in ((16, 1) if trial == 1 else (1, 16))
                            ],
                        }
                    )
        summary = summarize(results)
        for nodes in (1, 10):
            for backend, expected in (
                ("ort", 1),
                ("custom", 2),
                ("standalone", 0.5),
                ("native", 4),
            ):
                if (nodes, backend) not in summary:
                    self.assertEqual((nodes, backend), (10, "standalone"))
                    continue
                for row in summary[(nodes, backend)]:
                    self.assertEqual(row["speedup"], expected)
                    self.assertEqual(row["median_s"], 2 * nodes * row["size"] / expected)

        import matplotlib.pyplot as plt

        fig = plot_results(results)
        fig.canvas.draw()
        self.assertEqual(len(fig.axes), 4)
        self.assertEqual(len(fig.axes[0].lines), 4)
        self.assertEqual(len(fig.axes[1].lines), 3)
        for column in (0, 1):
            self.assertEqual(fig.axes[column].get_yscale(), "log")
            numpy.testing.assert_array_equal(fig.axes[column + 2].lines[0].get_ydata(), [1, 1])
            numpy.testing.assert_array_equal(fig.axes[column + 2].lines[1].get_ydata(), [2, 2])
        self.assertEqual(
            [text.get_text().replace("\n", " ") for text in fig.legends[0].texts],
            list(LABELS.values()),
        )
        plt.close(fig)

    def test_cli_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    plot_benchmark_log_custom_ops.__file__,
                    "--library",
                    str(LIBRARY),
                    "--host",
                    str(LIBRARY),
                    "--nodes",
                    "10",
                    "--sizes",
                    "33",
                    "7",
                    "--warmup",
                    "1",
                    "--repeat",
                    "2",
                    "--number",
                    "2",
                    "--trials",
                    "2",
                    "--output",
                    str(output),
                ],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["baseline"], LABELS["ort"])
            self.assertEqual(
                {r["backend"] for r in report["results"]}, {"ort", "custom", "native"}
            )
            self.assertEqual(len(report["results"]), 6)
            self.assertIn("Speedup vs ORT", completed.stdout)
            self.assertIn("unavailable (single-node host)", completed.stdout)
            for (_, backend), rows in summarize(report["results"]).items():
                for row in rows:
                    self.assertIn(
                        f"| {row['size']:,} | {LABELS[backend]} | "
                        f"{row['median_s'] * 1e6:.3f} | {row['speedup']:.2f}x |",
                        completed.stdout,
                    )

    def test_calling_paths(self):
        for nodes in (1, 10):
            for backend in ("ort", "custom"):
                with self.subTest(nodes=nodes, backend=backend):
                    result = measure(
                        SimpleNamespace(
                            backend=backend,
                            nodes=nodes,
                            library=LIBRARY,
                            sizes=[1, 7, 33],
                            warmup=1,
                            repeat=2,
                            number=2,
                        )
                    )
                    self.assertEqual(result["nodes"], nodes)
                    self.assertEqual(result["api"], "iobinding")
                    self.assertEqual([row["size"] for row in result["rows"]], [1, 7, 33])
                    for row in result["rows"]:
                        self.assertEqual(len(row["samples_s"]), 2)
                        self.assertTrue(all(sample > 0 for sample in row["samples_s"]))
        self.assertNotIn("onnx", sys.modules)

    def test_standalone_rejects_unsupported_topology(self):
        with self.assertRaisesRegex(ValueError, "only one node"):
            measure(SimpleNamespace(backend="standalone", nodes=10))

    def test_native_without_ort(self):
        for nodes in (1, 10):
            with self.subTest(nodes=nodes):
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys\n"
                        "from types import SimpleNamespace\n"
                        "from docs.examples.plot_benchmark_log_custom_ops import measure\n"
                        "from onnx_light_cpu import (\n"
                        "    set_kernel_usage_recording, used_kernel_names)\n"
                        "from onnx_light.onnx.reference import ReferenceEvaluator\n"
                        "from docs.examples.plot_benchmark_log_custom_ops import make_model\n"
                        "import numpy\n"
                        f"nodes = {nodes}\n"
                        "result = measure(SimpleNamespace(backend='native', nodes=nodes,\n"
                        "    sizes=[1, 7, 33], warmup=1, repeat=2, number=2))\n"
                        "assert result['api'] == 'run'\n"
                        "assert result['ort_version'] is None\n"
                        "assert result['nodes'] == nodes\n"
                        "assert [row['size'] for row in result['rows']] == [1, 7, 33]\n"
                        "assert all(len(row['samples_s']) == 2 and\n"
                        "    all(s > 0 for s in row['samples_s']) for row in result['rows'])\n"
                        "session = ReferenceEvaluator(make_model(False, nodes))\n"
                        "set_kernel_usage_recording(session, True)\n"
                        "session.run(None, {'X': numpy.ones(7, dtype=numpy.float32)})\n"
                        "assert used_kernel_names(session) == ['onnx_light_cpu::Log'] * nodes\n"
                        "assert 'onnxruntime' not in sys.modules\n"
                        "assert 'onnx_light_ort' not in sys.modules\n"
                        "assert 'onnx' not in sys.modules\n",
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True,
                )
                self.assertEqual(completed.stdout, "")

    def test_numpy_is_not_a_backend(self):
        with self.assertRaisesRegex(ValueError, "Unsupported backend"):
            measure(SimpleNamespace(backend="numpy", nodes=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    LIBRARY = arguments.library.resolve(strict=True)
    unittest.main(argv=[__file__, *remaining], verbosity=2)
