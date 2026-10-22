"""Checks the gallery's graph topologies and synchronous ORT calling paths."""

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

from docs.examples.plot_benchmark_log_custom_ops import make_model, measure, numpy_log
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
                    "numpy",
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
            self.assertEqual(result["backend"], "numpy")
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

    def test_numpy_chain(self):
        values = numpy.array([0.01, 1, 10], dtype=numpy.float32)
        original = values.copy()
        numpy.testing.assert_array_equal(numpy_log(values, 1), numpy.log(values))
        with numpy.errstate(divide="ignore", invalid="ignore"):
            self.assertTrue(numpy.isnan(numpy_log(values, 10)).all())
        numpy.testing.assert_array_equal(values, original)

    def test_calling_paths(self):
        for nodes in (1, 10):
            for backend in ("numpy", "ort", "custom"):
                apis = (
                    ("run",)
                    if backend == "numpy"
                    else ("run", "ortvalues", "iobinding", "ortvaluevector")
                )
                for api in apis:
                    with self.subTest(nodes=nodes, backend=backend, api=api):
                        result = measure(
                            SimpleNamespace(
                                backend=backend,
                                api=api,
                                nodes=nodes,
                                library=LIBRARY,
                                sizes=[1, 7, 33],
                                warmup=1,
                                repeat=2,
                                number=2,
                            )
                        )
                        self.assertEqual(result["nodes"], nodes)
                        self.assertEqual(result["api"], api)
                        self.assertEqual([row["size"] for row in result["rows"]], [1, 7, 33])
                        for row in result["rows"]:
                            self.assertEqual(len(row["samples_s"]), 2)
                            self.assertTrue(all(sample > 0 for sample in row["samples_s"]))
        self.assertNotIn("onnx", sys.modules)

    def test_standalone_rejects_unsupported_topology(self):
        for nodes, api in ((10, "run"), (1, "iobinding")):
            with self.subTest(nodes=nodes, api=api):
                with self.assertRaisesRegex(ValueError, "only one node"):
                    measure(SimpleNamespace(backend="standalone", nodes=nodes, api=api))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    LIBRARY = arguments.library.resolve(strict=True)
    unittest.main(argv=[__file__, *remaining], verbosity=2)
