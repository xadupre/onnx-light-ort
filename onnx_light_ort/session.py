"""Runs audited ORT custom-op libraries through the native onnx-light runtime."""

import ctypes
from pathlib import Path
import threading
import weakref

import numpy
from ml_dtypes import bfloat16
from onnx_light.onnx import ModelProto, TensorProto

_MAX_MODEL_BYTES = 16 * 1024 * 1024
_MAX_TENSOR_BYTES = 64 * 1024 * 1024
_MAX_RANK = 16


class CustomOpSession:
    """Owns a single-node floating-point custom-op session without loading ORT.

    Libraries execute trusted native code. Only the API-18 CPU subset documented
    in ``docs/components/bridge_ops_ort_called_by_light.rst`` is supported.
    """

    def __init__(self, model, *, library, host_library):
        if isinstance(model, bytes):
            serialized = model
        elif isinstance(model, ModelProto):
            serialized = model.SerializeToString()
        else:
            raise TypeError("model must be an onnx-light ModelProto or serialized bytes")
        if len(serialized) > _MAX_MODEL_BYTES:
            raise ValueError("Custom-op model must not exceed 16 MiB")
        if isinstance(model, bytes):
            model = ModelProto()
            model.ParseFromString(serialized)
        self._lock = threading.Lock()
        self._api = ctypes.CDLL(str(Path(host_library).resolve(strict=True)))
        self._api.LightOrtCreateSession.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._api.LightOrtCreateSession.restype = ctypes.c_void_p
        self._api.LightOrtRunTyped.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._api.LightOrtRunTyped.restype = ctypes.c_void_p
        self._api.LightOrtReleaseSession.argtypes = [ctypes.c_void_p]
        self._api.LightOrtReleaseSession.restype = None
        self._api.LightOrtGetErrorMessage.argtypes = [ctypes.c_void_p]
        self._api.LightOrtGetErrorMessage.restype = ctypes.c_char_p
        self._api.LightOrtReleaseStatus.argtypes = [ctypes.c_void_p]
        self._api.LightOrtReleaseStatus.restype = None
        self._input_name = model.graph.input[0].name if model.graph.input else None
        self._output_name = model.graph.output[0].name if model.graph.output else None
        handle = ctypes.c_void_p()
        self._check(
            self._api.LightOrtCreateSession(
                str(Path(library).resolve(strict=True)).encode("utf-8"),
                serialized,
                len(serialized),
                ctypes.byref(handle),
            )
        )
        self._handle = handle
        self._finalizer = weakref.finalize(self, self._api.LightOrtReleaseSession, handle)
        self._element_type = model.graph.input[0].type.tensor_type.elem_type
        self._dtype = {
            TensorProto.FLOAT: numpy.dtype(numpy.float32),
            TensorProto.DOUBLE: numpy.dtype(numpy.float64),
            TensorProto.FLOAT16: numpy.dtype(numpy.float16),
            TensorProto.BFLOAT16: numpy.dtype(bfloat16),
        }[self._element_type]

    def _check(self, status):
        if status:
            message = self._api.LightOrtGetErrorMessage(status).decode("utf-8", errors="replace")
            self._api.LightOrtReleaseStatus(status)
            raise RuntimeError(message)

    def run(self, output_names, inputs):
        """Returns independent NumPy outputs, reusing the native kernel between runs."""
        with self._lock:
            if not self._finalizer.alive:
                raise RuntimeError("Custom-op session is closed")
            if set(inputs) != {self._input_name}:
                raise ValueError(f"Expected only input {self._input_name!r}")
            if output_names is not None and list(output_names) != [self._output_name]:
                raise ValueError(f"Expected only output {self._output_name!r}")
            values = inputs[self._input_name]
            if not isinstance(values, numpy.ndarray) or values.dtype != self._dtype:
                raise TypeError(f"Custom-op input must be a {self._dtype.name} NumPy array")
            if values.ndim > _MAX_RANK:
                raise ValueError("Custom-op input supports at most 16 dimensions")
            if values.nbytes > _MAX_TENSOR_BYTES:
                raise ValueError("Custom-op input must not exceed 64 MiB")
            # The ORT API exposes mutable input pointers; protect the caller's array.
            values = numpy.array(values, dtype=self._dtype, order="C", copy=True)
            shape = (ctypes.c_int64 * values.ndim)(*values.shape)
            output = ctypes.c_void_p()
            dimensions = ctypes.POINTER(ctypes.c_int64)()
            rank = ctypes.c_size_t()
            count = ctypes.c_size_t()
            self._check(
                self._api.LightOrtRunTyped(
                    self._handle,
                    values.ctypes.data,
                    self._element_type,
                    values.size,
                    shape,
                    values.ndim,
                    ctypes.byref(output),
                    ctypes.byref(dimensions),
                    ctypes.byref(rank),
                    ctypes.byref(count),
                )
            )
            output_shape = tuple(dimensions[index] for index in range(rank.value))
            if count.value:
                storage = (ctypes.c_ubyte * (count.value * self._dtype.itemsize)).from_address(
                    output.value
                )
                result = numpy.frombuffer(storage, dtype=self._dtype).copy()
            else:
                result = numpy.empty(0, dtype=self._dtype)
            return [result.reshape(output_shape)]

    def close(self):
        """Destroys native kernels before unloading their custom-op library."""
        with self._lock:
            self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
