"""Registers onnx-light custom operators with an optional ONNX Runtime host."""

from pathlib import Path

from .libraries import get_library_path


def register_custom_ops_light_called_by_ort(options=None, *, library=None):
    """Registers onnx-light custom ops in ONNX Runtime's SessionOptions.

    ONNX Runtime is the host: this calls its ``register_custom_ops_library``.
    This does not register ORT custom ops in the onnx-light runtime.
    Existing options are modified in place without changing their settings.
    ``library`` can select another trusted native custom-op library.
    ONNX Runtime is imported only when this function is called.
    """
    import onnxruntime

    if options is None:
        options = onnxruntime.SessionOptions()
    path = (
        get_library_path("custom_ops") if library is None else Path(library).resolve(strict=True)
    )
    options.register_custom_ops_library(str(path))
    return options
