"""Compare steady-state CPU Log custom-op inference against standard ORT Log."""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy
from onnx_light.onnx import TensorProto, helper


def make_model(custom):
    domain = "onnx_light_cpu" if custom else ""
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("Log", ["X"], ["Y"], domain=domain)],
            "log-custom-op-benchmark",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N"])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N"])],
        ),
        opset_imports=[helper.make_opsetid(domain, 1 if custom else 18)],
        ir_version=9,
    ).SerializeToString()


def measure(args):
    custom = args.backend != "ort"
    if args.backend == "standalone":
        from onnx_light_ort import CustomOpSession

        session = CustomOpSession(
            make_model(custom), library=args.library, host_library=args.host
        )
        version = None
    else:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        if custom:
            options.register_custom_ops_library(str(args.library))
        session = onnxruntime.InferenceSession(
            make_model(custom), options, providers=["CPUExecutionProvider"]
        )
        version = onnxruntime.__version__

    rng = numpy.random.default_rng(42)
    rows = []
    for size in args.sizes:
        values = rng.uniform(0.01, 10, size).astype(numpy.float32)
        feeds = {"X": values}
        expected = numpy.log(values.astype(numpy.float64)).astype(numpy.float32)
        for _ in range(args.warmup):
            session.run(None, feeds)
        numpy.testing.assert_allclose(session.run(None, feeds)[0], expected, rtol=2e-6, atol=0)
        samples = []
        for _ in range(args.repeat):
            begin = time.perf_counter()
            for _ in range(args.number):
                session.run(None, feeds)
            samples.append((time.perf_counter() - begin) / args.number)
        numpy.testing.assert_allclose(session.run(None, feeds)[0], expected, rtol=2e-6, atol=0)
        rows.append({"size": size, "samples_s": samples})

    if args.backend == "standalone":
        session.close()
        assert "onnxruntime" not in sys.modules
    assert "onnx" not in sys.modules
    return {"backend": args.backend, "ort_version": version, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1024, 32768, 65536, 262144, 1048576]
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=31)
    parser.add_argument("--number", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--backend", choices=["ort", "custom", "standalone"], help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if min(args.sizes) <= 0 or max(args.sizes) > 16 * 1024 * 1024:
        parser.error("sizes must be positive and fit the host's 64 MiB tensor limit")
    if min(args.repeat, args.number, args.trials) <= 0 or args.warmup < 0:
        parser.error("repeat, number and trials must be positive; warmup must be nonnegative")
    args.library = args.library.resolve(strict=True)
    args.host = args.host.resolve(strict=True)
    if args.backend:
        print(json.dumps(measure(args)))
        return

    results = []
    for trial in range(args.trials):
        backends = ["ort", "custom", "standalone"]
        backends = backends[trial % 3 :] + backends[: trial % 3]
        for backend in backends:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--library",
                str(args.library),
                "--host",
                str(args.host),
                "--sizes",
                *(str(size) for size in args.sizes),
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(args.repeat),
                "--number",
                str(args.number),
                "--backend",
                backend,
            ]
            completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
            results.append(json.loads(completed.stdout))

    report = {
        "measurement": "steady-state session.run; includes allocation and Python interfaces",
        "threads": 1,
        "operator": "Log",
        "dtype": "float32",
        "input_distribution": "uniform(0.01, 10), NumPy default_rng seed 42",
        "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": platform.python_version(),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "number": args.number,
        "trials": args.trials,
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "| Elements | ORT Log (us) | CPU custom op / ORT (us) | Speedup | "
        "CPU custom op / standalone (us) |"
    )
    print("| ---: | ---: | ---: | ---: | ---: |")
    for index, size in enumerate(args.sizes):
        times = {}
        for backend in ("ort", "custom", "standalone"):
            samples = [
                value
                for result in results
                if result["backend"] == backend
                for value in result["rows"][index]["samples_s"]
            ]
            times[backend] = statistics.median(samples)
        print(
            f"| {size:,} | {times['ort'] * 1e6:.3f} | {times['custom'] * 1e6:.3f} "
            f"| {times['ort'] / times['custom']:.2f}x "
            f"| {times['standalone'] * 1e6:.3f} |"
        )
    print(
        "Raw samples, when --output is supplied, include all trials; "
        "standalone timings include protective input and output copies."
    )


if __name__ == "__main__":
    main()
