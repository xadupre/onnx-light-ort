# Common support for hosting ORT-compatible plugins

`onnx_light_ort::common_ort_to_light` is a static library implementing the
audited API-18 CPU custom-op subset: statuses, registration domains, float
attributes, tensor/context adapters, shapes and output allocation.
It never installs, loads or links ORT and is independent of the outgoing
common library. Unsupported table entries are null; EP support is still planned.

The dependency-contract smoke test remains, and the incoming bridge adds
independent custom-op library execution and API-version failure tests.
Run `ctest --test-dir build -L common_ort_to_light --output-on-failure`.
See [the component documentation](../../docs/components/common_ort_to_light.rst)
for the exact supported function inventory and profile limits.
