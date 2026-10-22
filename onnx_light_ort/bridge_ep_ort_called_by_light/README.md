# EP bridge: ORT plugins called by onnx-light

Defines `onnx_light_ort::bridge_ep_ort_called_by_light`, currently an interface target
depending on `common_ort_to_light`. It reserves foreign EP factory/device
lifetimes, provider selection and kernel/partition registration in onnx-light.

It must never install, load or link ONNX Runtime. Shared API-table, tensor and
lifetime machinery belongs in its directional common library, not the ops
bridge. Runtime code and executable unit tests are not implemented yet.
See [the component page](../../docs/components/bridge_ep_ort_called_by_light.rst).
