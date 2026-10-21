Custom execution provider
=========================

Reserved for the ORT-loadable execution-provider library. It will own the
``CreateEpFactories`` and ``ReleaseEpFactory`` exports and provider callbacks,
using ``bridge_light_to_ort`` for execution and tensor adaptation.

This component has no implementation or CMake target yet. It must remain
independent of the custom-op library and the standalone compatibility host.
See ``docs/architecture.rst`` and ``docs/next_steps.rst`` for the design.
