"""Locates bundled native libraries or the matching source-tree build."""

import json
from pathlib import Path
import sys


def _get_build_library_path(build, component):
    configuration = {}
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        name = key.partition(":")[0]
        if separator and name in {"CMAKE_BUILD_TYPE", "CMAKE_CONFIGURATION_TYPES"}:
            configuration[name] = value
    config = (
        "Release"
        if configuration.get("CMAKE_CONFIGURATION_TYPES")
        else configuration.get("CMAKE_BUILD_TYPE", "")
    )
    manifest = build / config / "python_libraries.json"
    libraries = json.loads(manifest.read_text(encoding="utf-8"))
    if component not in libraries:
        raise FileNotFoundError(f"Native component {component!r} is disabled in {manifest}")
    return Path(libraries[component]).resolve(strict=True)


def get_library_path(component="custom_ops"):
    """Returns an absolute library path from a wheel or the source build.

    Source builds use CMake's generated target manifest in the existing ``build``
    tree: the active single-config build type, or Release for multi-config
    generators. No system directories or model-supplied paths are searched.
    """
    names = {
        "custom_ops": "onnx_light_ort_custom_ops",
        "host": "onnx_light_ort_bridge_ops_ort_called_by_light",
    }
    if component not in names:
        raise ValueError(f"Unknown native component {component!r}; expected custom_ops or host")
    if sys.platform == "win32":
        filename = names[component] + ".dll"
    else:
        filename = "lib" + names[component] + (".dylib" if sys.platform == "darwin" else ".so")
    package = Path(__file__).resolve().parent
    native = package / "_native"
    if native.is_dir():
        return (native / filename).resolve(strict=True)
    root = package.parent
    if (root / "CMakeLists.txt").is_file() and (root / "build" / "CMakeCache.txt").is_file():
        return _get_build_library_path(root / "build", component)
    raise FileNotFoundError(
        f"Native component {component!r} is not built. From the source checkout, run "
        "'python setup.py build_ext --inplace', or install a release wheel."
    )
