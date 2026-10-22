---
name: onnx-light-only
description: Use when creating, loading, checking or serializing models, or changing Python tests, examples, dependencies or CI in onnx-light-ort. Use onnx-light, never the onnx Python package.
---

# Use onnx-light, not onnx

- Do not import, install or add a dependency on the `onnx` Python package,
  including optional imports, fallbacks or aliases in `sys.modules`.
- Use `onnx_light.onnx` model classes, helpers, checker and serialization APIs.
  For serialized models, construct `ModelProto` and call `ParseFromString`.
- Build matching onnx-light Python model bindings from the fetched source in
  the repository's existing build tree. Use `build/python` through `PYTHONPATH`;
  do not install local project artifacts into site-packages or use editable
  installs.
- Keep Python model construction and native execution on the same onnx-light
  source revision. Do not introduce another compiled onnx-light version.
- ONNX is still the model format. ONNX Runtime is permitted only for the
  explicit real-ORT integration direction; it must not be imported or loaded by
  the standalone incoming host.
- Give custom-op example files names containing `custom_ops`. Their titles and
  descriptions must state that they demonstrate custom operators, not EPs.
- Name bridges `bridge_{ops,ep}_{implementation}_called_by_{host}` so the caller
  is explicit, for example `bridge_ops_ort_called_by_light`.
- Validate both custom-op directions and execute the gallery. Preserve checks
  that the `onnx` package is not imported; do not skip missing dependencies.
