"""Checks tag and metadata validation without loading native dependencies."""

from pathlib import Path
import runpy
import shutil
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
RELEASE_HELPERS = runpy.run_path(str(ROOT / ".github" / "scripts" / "check_release_version.py"))
CHECK_VERSION = RELEASE_HELPERS["check_version"]
PIN_REVISION = RELEASE_HELPERS["pin_onnx_light_revision"]


class TestReleaseVersion(unittest.TestCase):
    def test_dependency_release_versions(self):
        source = (ROOT / "cmake" / "Dependencies.cmake").read_text(encoding="utf-8")
        self.assertRegex(
            source, r'set\(ONNX_LIGHT_ORT_ONNX_LIGHT_REVISION "0\.2\.1" CACHE STRING'
        )
        self.assertRegex(
            source,
            r"GIT_REPOSITORY https://github\.com/xadupre/onnx-light-cpu\.git\s+"
            r"GIT_TAG 0\.2\.1\.1\s",
        )
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(
            metadata["project"]["optional-dependencies"]["models"], ["onnx-light==0.2.1"]
        )

    def test_dependency_pinning_preserves_formatting(self):
        for assignment in (
            '  set(ONNX_LIGHT_ORT_ONNX_LIGHT_REVISION "main" CACHE STRING "description")',
            'set ( ONNX_LIGHT_ORT_ONNX_LIGHT_REVISION\n  "main"\n CACHE STRING\n "new text" )',
            'set(ONNX_LIGHT_ORT_ONNX_LIGHT_REVISION main CACHE STRING "description")',
        ):
            with self.subTest(assignment=assignment):
                source = f"# Keep this comment.\n{assignment}\n# Keep this too.\n"
                expected = source.replace('"main"', '"abc123"').replace(" main ", ' "abc123" ')
                self.assertEqual(PIN_REVISION(source, "abc123"), expected)
                self.assertEqual(PIN_REVISION(expected, "abc123"), expected)
                self.assertNotEqual(PIN_REVISION(expected, "def456"), expected)

    def test_dependency_pinning_requires_one_assignment(self):
        assignment = 'set(ONNX_LIGHT_ORT_ONNX_LIGHT_REVISION "main" CACHE STRING "")\n'
        for source in ("", "# " + assignment, assignment * 2):
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "Expected one cached"):
                    PIN_REVISION(source, "abc123")

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
