"""Checks tag and metadata validation without loading native dependencies."""

from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
RELEASE_HELPERS = runpy.run_path(str(ROOT / ".github" / "scripts" / "check_release_version.py"))
CHECK_VERSION = RELEASE_HELPERS["check_version"]
PIN_REVISION = RELEASE_HELPERS["pin_onnx_light_revision"]
DEPENDENCY_VERSIONS = runpy.run_path(
    str(ROOT / ".github" / "scripts" / "dependency_versions.py")
)["dependency_versions"]


class TestReleaseVersion(unittest.TestCase):
    def test_dependency_release_versions(self):
        source = (ROOT / "cmake" / "Dependencies.cmake").read_text(encoding="utf-8")
        self.assertRegex(
            source,
            r"set\(ONNX_LIGHT_ORT_ONNX_LIGHT_REVISION "
            r'"\$\{ONNX_LIGHT_ORT_ONNX_LIGHT_VERSION\}" CACHE STRING',
        )
        self.assertRegex(
            source,
            r"GIT_REPOSITORY https://github\.com/xadupre/onnx-light-cpu\.git\s+"
            r'GIT_TAG "\$\{ONNX_LIGHT_ORT_ONNX_LIGHT_CPU_VERSION\}"\s',
        )
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        versions = DEPENDENCY_VERSIONS(ROOT)
        self.assertEqual(versions, {"onnx-light": "0.2.5", "onnx-light-cpu": "0.2.5"})
        self.assertEqual(
            metadata["project"]["optional-dependencies"]["models"],
            [f"onnx-light=={versions['onnx-light']}"],
        )
        self.assertEqual(metadata["tool"]["scikit-build"]["cmake"]["version"], ">=4.0")
        self.assertIn(
            "cmake>=4.0",
            (ROOT / "docs" / "requirements.txt").read_text(encoding="utf-8").splitlines(),
        )
        self.assertIn(
            "cmake_minimum_required(VERSION 4.0)",
            (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"),
        )

    def test_workflow_dependency_release_versions(self):
        workflows = ROOT / ".github" / "workflows"
        for name in ("ci.yml", "coverage.yml", "docs.yml", "fuzz.yml"):
            with self.subTest(workflow=name):
                source = (workflows / name).read_text(encoding="utf-8")
                revisions = re.findall(r"repository: xadupre/onnx-light\s+ref: ([^\n]+)", source)
                self.assertTrue(revisions)
                self.assertEqual(set(revisions), {"${{ steps.dependencies.outputs.onnx_light }}"})
                self.assertEqual(source.count("id: dependencies"), len(revisions))
                self.assertEqual(
                    source.count(
                        'python .github/scripts/dependency_versions.py >> "$GITHUB_OUTPUT"'
                    ),
                    len(revisions),
                )
        for name in ("releases.yml", "release-python.yml"):
            with self.subTest(workflow=name):
                source = (workflows / name).read_text(encoding="utf-8")
                self.assertIn("python .github/scripts/dependency_versions.py onnx-light", source)
                self.assertNotRegex(source, r"repos/xadupre/onnx-light/commits/[0-9]")

    def test_dependency_versions_cli(self):
        script = ROOT / ".github" / "scripts" / "dependency_versions.py"
        for dependency in (None, "onnx-light", "onnx-light-cpu"):
            with self.subTest(dependency=dependency):
                command = [sys.executable, str(script)]
                if dependency:
                    command.append(dependency)
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                expected = (
                    DEPENDENCY_VERSIONS(ROOT)[dependency] + "\n"
                    if dependency
                    else "onnx_light=0.2.5\nonnx_light_cpu=0.2.5\n"
                )
                self.assertEqual(result.stdout, expected)

    def test_cmake_dependency_versions(self):
        for light, cpu in (("0.2.5", "0.2.5"), ("1.2.3", "4.5.6"), ("bad", "0.2.5")):
            with self.subTest(light=light, cpu=cpu), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "cmake").mkdir()
                shutil.copyfile(
                    ROOT / "cmake" / "DependencyVersions.cmake",
                    root / "cmake" / "DependencyVersions.cmake",
                )
                metadata = (
                    "[tool.onnx-light-ort.dependencies]\n"
                    f'onnx-light = "{light}"\nonnx-light-cpu = "{cpu}"\n'
                    '[tool.other]\nonnx-light = "9.9.9"\n'
                )
                (root / "pyproject.toml").write_text(metadata, encoding="utf-8")
                script = root / "versions.cmake"
                script.write_text(
                    'include("${CMAKE_CURRENT_LIST_DIR}/cmake/DependencyVersions.cmake")\n'
                    'file(WRITE "${CMAKE_CURRENT_LIST_DIR}/versions.txt" '
                    '"${ONNX_LIGHT_ORT_ONNX_LIGHT_VERSION}\\n'
                    '${ONNX_LIGHT_ORT_ONNX_LIGHT_CPU_VERSION}\\n")\n',
                    encoding="utf-8",
                )
                result = subprocess.run(
                    ["cmake", "-P", str(script)], capture_output=True, text=True
                )
                if light == "bad":
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Expected one onnx-light version", result.stderr)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(
                        (root / "versions.txt").read_text(encoding="utf-8"),
                        f"{light}\n{cpu}\n",
                    )
                    self.assertEqual(
                        DEPENDENCY_VERSIONS(root), {"onnx-light": light, "onnx-light-cpu": cpu}
                    )

    def test_dependency_pinning_preserves_formatting(self):
        for assignment in (
            '  set(ONNX_LIGHT_ORT_ONNX_LIGHT_REVISION "main" CACHE STRING "description")',
            'set ( ONNX_LIGHT_ORT_ONNX_LIGHT_REVISION\n  "main"\n CACHE STRING\n "new text" )',
            'set(ONNX_LIGHT_ORT_ONNX_LIGHT_REVISION main CACHE STRING "description")',
            "set(ONNX_LIGHT_ORT_ONNX_LIGHT_REVISION "
            '"${ONNX_LIGHT_ORT_ONNX_LIGHT_VERSION}" CACHE STRING "description")',
        ):
            with self.subTest(assignment=assignment):
                source = f"# Keep this comment.\n{assignment}\n# Keep this too.\n"
                expected = source.replace('"main"', '"abc123"').replace(" main ", ' "abc123" ')
                expected = expected.replace('"${ONNX_LIGHT_ORT_ONNX_LIGHT_VERSION}"', '"abc123"')
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
