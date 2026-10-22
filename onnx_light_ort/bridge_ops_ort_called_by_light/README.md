# Custom-op bridge: ORT custom ops called by onnx-light

`onnx_light_ort::bridge_ops_ort_called_by_light` is a shared host using the directional
`common_ort_to_light` static library and the platform loader, never ORT.
The source-checkout `CustomOpSession` Python wrapper runs audited API-18 CPU
plugins in a real onnx-light `RuntimeSession`.

The initial profile supports single-node models with one required float16,
bfloat16, float32 or float64 input/output, matching model and buffer types,
dynamic/scalar/empty shapes, float attributes and V2 callbacks.
Typed overloads are selected from the model dtype. `LightOrtRunTyped` handles
all four types; the existing `LightOrtRun` ABI remains float32-only.
It is not a general compatibility layer and does not implement EPs.
Plugins execute trusted code; audit their required API subset before loading.
The `onnx_light_cpu::Log` plugin also fits this profile and runs through the
same generic host without loading ORT or linking the outgoing bridge.

Native custom-op tests and optional Python integration tests cover execution,
errors and kernel reuse/teardown. Select `bridge_ops_ort_called_by_light` with CTest.
See [the component page](../../docs/components/bridge_ops_ort_called_by_light.rst).
