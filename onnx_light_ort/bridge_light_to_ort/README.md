# onnx-light to ORT bridge

Builds the static library `onnx_light_ort::bridge_light_to_ort`, used by the
custom-op library and intended for the independent future EP library.
It retains native kernel state and adapts invocation-scoped ORT inputs and
outputs without copying tensor payloads.

Native unit tests live in `tests/`. After a normal build, run
`ctest --test-dir build -L bridge_light_to_ort --output-on-failure`.
They exercise the real adapter and numerical kernel with an explicit minimal
C API host, including failures that real ORT normally rejects before dispatch.
No mocking library or installed ORT runtime is needed.

See [the component documentation](../../docs/components/bridge_light_to_ort.rst)
for buffer validation, exception propagation and lifetime requirements.
