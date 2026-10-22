# Custom execution provider

Reserved for an independent ORT-loadable execution-provider library owning
`CreateEpFactories`, `ReleaseEpFactory` and provider callbacks. It will use
`bridge_ep_light_called_by_ort` and its `common_light_to_ort` dependency, not depend
on the custom-op library, the ops bridge or the opposite
compatibility bridge.

This component has no implementation, CMake target or executable tests yet.
Factory/device lifecycle, node selection, unsupported-node fallback, kernel
state and teardown tests must accompany the first EP implementation; testing
empty declarations would not validate an EP.

See [the component documentation](../../docs/components/custom_ep.rst) and
the [roadmap](../../docs/newt_steps/2026/2026-10-start.rst).
