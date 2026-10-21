# ORT to onnx-light bridge

Defines `onnx_light_ort::bridge_ort_to_light`, currently an interface target.
The planned implementation hosts independent ORT-compatible custom ops and EPs
inside onnx-light without installing, loading or linking ONNX Runtime.

The tests in `tests/` are dependency-contract smoke tests, not tests of a
working compatibility host. They serialize custom-domain graphs and execute
a real onnx-light custom kernel across changing shapes.
Run `ctest --test-dir build -L bridge_ort_to_light --output-on-failure`.

API-table, plugin-loading and ownership tests will accompany their respective
implementations. The explicit unit-test host used by the reverse bridge is
test infrastructure, not an implementation of this component.

See [the component documentation](../../docs/components/bridge_ort_to_light.rst).
