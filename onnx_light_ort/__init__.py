"""Python interoperability between onnx-light and ONNX Runtime.

Custom operators are supported in both directions. Execution-provider bridges
are not implemented yet.
"""

from .session import CustomOpSession, CustomOpSessionOrtCalledByLight
from .libraries import get_library_path
from .registration import register_custom_ops_light_called_by_ort

__version__ = "0.2.0"

__all__ = [
    "CustomOpSession",
    "CustomOpSessionOrtCalledByLight",
    "get_library_path",
    "register_custom_ops_light_called_by_ort",
]
