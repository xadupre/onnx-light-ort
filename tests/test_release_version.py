"""Checks tag and metadata validation without loading native dependencies."""

from pathlib import Path
import runpy
import shutil
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
CHECK_VERSION = runpy.run_path(str(ROOT / ".github" / "scripts" / "check_release_version.py"))[
    "check_version"
]


class TestReleaseVersion(unittest.TestCase):
    def test_matching_tags_and_dry_run(self):
        version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "version"
        ]
        for tag in ("", version, "v" + version):
            with self.subTest(tag=tag):
                self.assertEqual(CHECK_VERSION(ROOT, tag), version)

    def test_mismatched_tag(self):
        with self.assertRaisesRegex(ValueError, "does not match project version"):
            CHECK_VERSION(ROOT, "v9999.0.0")

    def test_mismatched_python_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("pyproject.toml", "CMakeLists.txt"):
                shutil.copyfile(ROOT / name, root / name)
            (root / "onnx_light_ort").mkdir()
            (root / "onnx_light_ort" / "__init__.py").write_text(
                '__version__ = "9999.0.0"\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Inconsistent project versions"):
                CHECK_VERSION(root, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
