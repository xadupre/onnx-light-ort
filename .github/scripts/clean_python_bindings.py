"""Remove previous CPython model extensions before reusing the wheel build tree."""

from pathlib import Path

directory = Path(__file__).resolve().parents[2] / "build" / "python" / "onnx_light" / "onnx_py"
for path in directory.glob("_onnxpyproto*"):
    if path.is_file() and path.suffix in {".so", ".pyd"}:
        path.unlink()
