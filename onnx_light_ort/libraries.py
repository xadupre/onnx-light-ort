"""Locates the native libraries bundled in a release wheel."""

from pathlib import Path
import sys


def get_library_path(component="custom_ops"):
    """Returns the installed host or custom-op library; source builds use explicit paths."""
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
    return (Path(__file__).parent / "_native" / filename).resolve(strict=True)
