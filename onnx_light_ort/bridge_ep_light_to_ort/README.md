# Execution providers: onnx-light in ORT

Defines `onnx_light_ort::bridge_ep_light_to_ort`, currently an interface target
depending on `common_light_to_ort`. It reserves EP-specific kernel-registry,
capability and execution adapters for the future `custom_ep` library.

It does not depend on `bridge_ops_light_to_ort` or the incoming compatibility
layers. EP runtime code and executable tests are not implemented yet.
See [the component page](../../docs/components/bridge_ep_light_to_ort.rst).
