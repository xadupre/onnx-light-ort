"""Registers onnx-light custom operators with an optional ONNX Runtime host."""

from pathlib import Path

from .libraries import get_library_path


def register_custom_ops(options=None, *, library=None):
    """Registers the bundled plugin and returns ORT SessionOptions.

    Existing options are modified in place without changing their settings.
    ``library`` can select another trusted native custom-op library.
    ONNX Runtime is imported only when this function is called.
    """
    import onnxruntime

    if options is None:
        options = onnxruntime.SessionOptions()
    path = get_library_path() if library is None else Path(library).resolve(strict=True)
    options.register_custom_ops_library(str(path))
    return options
