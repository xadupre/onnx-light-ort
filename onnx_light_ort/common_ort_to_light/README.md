# Common support for hosting ORT-compatible plugins

Defines `onnx_light_ort::common_ort_to_light`, currently an interface target.
The planned implementation shares API tables, opaque-value adapters and
resource lifetimes between `bridge_ops_ort_to_light` and
`bridge_ep_ort_to_light`, without installing, loading or linking ONNX Runtime.
The bridges own their profile-specific loading and registration.

The tests in `tests/` are dependency-contract smoke tests, not tests of a
working compatibility host. They serialize custom-domain graphs and execute
a real onnx-light custom kernel across changing shapes.
Run `ctest --test-dir build -L common_ort_to_light --output-on-failure`.

API-table, plugin-loading and ownership tests will accompany their respective
implementations. The explicit unit-test host used by the reverse bridge is
test infrastructure, not an implementation of this component.

See [the component documentation](../../docs/components/common_ort_to_light.rst).
