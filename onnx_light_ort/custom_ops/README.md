# Custom ops

Builds `onnx_light_ort::custom_ops`, an ORT-loadable shared library exporting
`RegisterCustomOps`. It registers float32 CPU `onnx_light::LeakyRelu` at custom
opset 1, with a required float `alpha` attribute.

It uses `bridge_ops_light_to_ort`, backed by `common_light_to_ort`, never the
EP bridge or opposite-direction compatibility layers.
The library has no link dependency on ONNX Runtime.

From the repository root, build normally and run
`ctest --test-dir build -L custom_ops --output-on-failure`.
Native unit tests in `tests/` use an explicit minimal C API host, without a
mocking library or ORT installation. Opt-in integration tests run in real ORT.

See [the component documentation](../../docs/components/custom_ops.rst) for
the ownership contract and runnable
[C++](../../examples/custom_ops/custom_ops.cc) and
[Python](../../examples/custom_ops/custom_ops.py) examples.
