# Common support for onnx-light in ORT

Builds the static library `onnx_light_ort::common_light_to_ort`, used by the
two bridges `bridge_ops_light_called_by_ort` and `bridge_ep_light_called_by_ort`.
It retains native kernel state and adapts invocation-scoped ORT inputs and
outputs without copying tensor payloads. It also provides shared
exception-to-status conversion, but no profile-specific registration callbacks.

Native unit tests live in `tests/`. After a normal build, run
`ctest --test-dir build -L common_light_to_ort --output-on-failure`.
They exercise the real adapter and numerical kernel with an explicit minimal
C API host, including failures that real ORT normally rejects before dispatch.
No mocking library or installed ORT runtime is needed.

See [the component documentation](../../docs/components/common_light_to_ort.rst)
for buffer validation, exception propagation and lifetime requirements.
