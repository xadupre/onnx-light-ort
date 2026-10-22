"""Runs the actual CPU plugin through the standalone host, without ORT."""

import argparse
from pathlib import Path
import sys
import unittest

from onnx_light_ort import CustomOpSession

from cpu_custom_ops_test_cases import CpuCustomOpCases, make_cpu_log_model

HOST = None
LIBRARY = None


class TestCpuCustomOpsOrtCalledByLight(CpuCustomOpCases, unittest.TestCase):
    input_errors = (TypeError, ValueError)

    def make_cpu_session(self):
        session = CustomOpSession(make_cpu_log_model(), library=LIBRARY, host_library=HOST)
        self.addCleanup(session.close)
        return session

    def tearDown(self):
        self.assertNotIn("onnxruntime", sys.modules)
        self.assertNotIn("onnx", sys.modules)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    HOST = arguments.host.resolve(strict=True)
    LIBRARY = arguments.library.resolve(strict=True)
    unittest.main(argv=[__file__, *remaining], verbosity=2)
