# Custom-op bridge: onnx-light called by ORT

Builds `onnx_light_ort::bridge_ops_light_called_by_ort`, a static library implementing
custom-op kernel construction, attribute handling, compute and destruction.
It depends on `common_light_to_ort` for native execution, tensor validation
and exception-to-status conversion. The `custom_ops` entry-point library wires
these callbacks into its operator descriptors.

Run `ctest --test-dir build -L bridge_ops_light_called_by_ort --output-on-failure`.
The native unit suite covers independent instances, explicit zero alpha and
exception/status handling; real-ORT integration tests also carry this label.

See [the component page](../../docs/components/bridge_ops_light_called_by_ort.rst).
