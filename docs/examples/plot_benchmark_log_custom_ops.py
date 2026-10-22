"""
Benchmark Log: built-in and custom operators
============================================

Compare synchronous CPU calls to ``Log`` using NumPy, ORT's built-in operator,
and the ``onnx_light_cpu`` custom operator (not an execution provider).
Two graph topologies are used: ``X -> Log -> Y`` and
``X -> Log -> ... -> Log -> Y`` with ten consecutive nodes. Both use float32,
dynamic one-dimensional inputs and models built with ``onnx_light.onnx``.

ORT executes each topology with either standard-domain or custom-domain nodes,
using all four synchronous Python calling interfaces: ``run``,
``run_with_ort_values``, ``run_with_iobinding`` and ``run_with_ortvaluevector``.
The standalone onnx-light custom-op host exposes only ``run`` and accepts only
one node. Its ten-node result is therefore unavailable, not ten Python calls
misrepresented as a graph. NumPy provides a baseline with one or ten ufunc calls.

Repeated real-valued logarithms inevitably leave the positive domain in float32.
The ten-node graph produces NaNs: this is a literal Log chain, not a benchmark
of ten finite-input logarithms. Every path is checked against a NumPy reference,
including matching NaNs, before and after timing.
"""

import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

from examples.custom_ops import benchmark_cpu_custom_ops

# %%
# Reuse the command-line benchmark's model construction and measurement code.
# Each measurement runs in a fresh process so the standalone host never imports
# or loads ORT. Library paths are explicit, just like the other gallery examples.
benchmark = Path(benchmark_cpu_custom_ops.__file__).resolve()
root = benchmark.parents[2]
environment = os.environ.copy()
environment["PYTHONPATH"] = os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")])
launcher = [sys.executable]
if sys.platform == "win32":
    launcher.append(str(root / "cmake/run_python_with_dll_paths.py"))
library = Path(os.environ["ONNX_LIGHT_ORT_PLUGIN_LIBRARY"]).resolve(strict=True)
host = Path(os.environ["ONNX_LIGHT_ORT_HOST_LIBRARY"]).resolve(strict=True)
sizes = [1, 1024, 65536]
apis = ["run", "ortvalues", "iobinding", "ortvaluevector"]
paths = [
    ("numpy", "run"),
    ("standalone", "run"),
    *((backend, api) for backend in ("ort", "custom") for api in apis),
]

# %%
# Sessions, random inputs (seed 42), OrtValues, bindings and preallocated outputs
# are prepared outside timing. ORT uses one thread and sequential execution.
# Each path gets three warmups and five batches of ten calls; report the median
# time per *whole graph*, not per node. Correctness checks and output conversion
# to NumPy are excluded from timing.
#
# ``run`` and ``run_with_ort_values`` allocate outputs; I/O binding and
# ``run_with_ortvaluevector`` reuse preallocated output buffers. The standalone
# wrapper includes protective input and output copies. These are end-to-end
# Python interface timings, not isolated kernel timings or universal speedups.
print(f"Python {platform.python_version()}, {platform.platform()}")
results = []
for nodes in (1, 10):
    print(f"\n{nodes} Log node(s), median microseconds per graph")
    print(f"{'Backend / API':<34}" + "".join(f"{size:>14,}" for size in sizes))
    for backend, api in paths:
        label = f"{backend} / {api}"
        if backend == "standalone" and nodes != 1:
            print(f"{label:<34}  unavailable (single-node host)")
            continue
        command = [
            *launcher,
            str(benchmark),
            "--library",
            str(library),
            "--host",
            str(host),
            "--sizes",
            *(str(size) for size in sizes),
            "--nodes",
            str(nodes),
            "--api",
            api,
            "--backend",
            backend,
            "--warmup",
            "3",
            "--repeat",
            "5",
            "--number",
            "10",
        ]
        completed = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, text=True, env=environment
        )
        result = json.loads(completed.stdout)
        results.append(result)
        times = [statistics.median(row["samples_s"]) * 1e6 for row in result["rows"]]
        print(f"{label:<34}" + "".join(f"{value:>14.3f}" for value in times))

# %%
# The full samples remain available in ``results``. No performance thresholds
# are asserted: CPU, SIMD dispatch and ORT version affect these measurements.
ort_version = next(result["ort_version"] for result in results if result["backend"] == "ort")
print(f"\nONNX Runtime {ort_version}; NumPy and onnx-light model bindings; float32 CPU")
assert "onnx" not in sys.modules
