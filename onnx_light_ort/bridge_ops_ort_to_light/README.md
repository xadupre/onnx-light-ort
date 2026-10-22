# Custom ops: ORT-compatible plugins in onnx-light

Defines `onnx_light_ort::bridge_ops_ort_to_light`, currently an interface target
depending on `common_ort_to_light`. It reserves custom-op library loading,
`RegisterCustomOps` invocation and node/kernel registration in onnx-light.

It must never install, load or link ONNX Runtime. Shared API-table, tensor and
lifetime machinery belongs in its directional common library, not the EP
bridge. Runtime code and executable unit tests are not implemented yet.
See [the component page](../../docs/components/bridge_ops_ort_to_light.rst).
