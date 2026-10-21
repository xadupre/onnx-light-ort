# Custom execution provider

Reserved for an independent ORT-loadable execution-provider library owning
`CreateEpFactories`, `ReleaseEpFactory` and provider callbacks. It will use
`bridge_light_to_ort`, not depend on the custom-op library or the opposite
compatibility bridge.

This component has no implementation, CMake target or executable tests yet.
Factory/device lifecycle, node selection, unsupported-node fallback, kernel
state and teardown tests must accompany the first EP implementation; testing
empty declarations would not validate an EP.

See [the component documentation](../../docs/components/custom_ep.rst) and
the [roadmap](../../docs/next_steps.rst).
