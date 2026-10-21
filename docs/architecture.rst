Architecture
============

Four libraries, two directions
------------------------------

See :doc:`next_steps` for the detailed implementation roadmap and the
inventory of host functions and plugin callbacks.

Each component has its own directory under ``onnx_light_ort/``. The root
CMake file selects components and tests; it does not define a monolithic
plugin or keep implementation sources in a root ``src/`` directory.

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Component / target
     - Responsibility
     - Current status
   * - ``onnx_light_ort::custom_ops``
     - ORT-loadable library owning ``RegisterCustomOps``, operator descriptors,
       domains and custom-op callbacks.
     - Shared library.
   * - ``custom_ep`` (future target)
     - Independent ORT-loadable library owning ``CreateEpFactories``,
       ``ReleaseEpFactory`` and EP callbacks.
     - Reserved directory; not implemented.
   * - ``onnx_light_ort::bridge_light_to_ort``
     - Native kernel state and tensor/context adapters for running onnx-light
       kernels inside ORT, shared by the custom-op and future EP libraries.
     - Static library.
   * - ``onnx_light_ort::bridge_ort_to_light``
     - Standalone compatibility host: loads ORT-compatible custom ops and EPs
       into onnx-light and supplies their API tables without ORT.
     - Interface target only; runtime planned.

The dependency graph is deliberately directional:

.. code-block:: text

   custom_ops ----> bridge_light_to_ort ----> onnx-light
   custom_ep  ----> bridge_light_to_ort ----> onnx-light   (planned)
   bridge_ort_to_light --------------------> onnx-light   (planned runtime)

The custom-op and EP libraries must remain independently loadable. Neither
may link the opposite-direction bridge, its API tables or its bootstrap
exports. Provider-neutral helpers can be shared later, never the host shim.
No component links ORT. The real consuming application supplies ORT to
``custom_ops`` and ``custom_ep`` through its API tables; only their opt-in
integration tests need an installed runtime.

The existing ``onnx_light_ort::plugin`` alias maps to ``::custom_ops`` and
``onnx_light_ort::host`` maps to ``::bridge_ort_to_light``. The existing
``BUILD_PLUGIN`` option selects custom ops and their reverse bridge;
``BUILD_HOST`` selects the standalone bridge placeholder. These names do not
merge the planned EP into the custom-op library.

The dependency test compiles
ORT's public C/EP declarations alongside onnx-light headers, links onnx-light's
core, round-trips a custom-domain model, and executes a native custom kernel
with changing shapes, including scalar and empty tensors. A separate optional
test loads the plugin in real ORT; the default tests do not load ORT.

Implemented custom-op profile
-----------------------------

The shared library registers ``onnx_light::LeakyRelu`` at custom-domain opset 1.
It advertises one required float32 CPU input and output and requires a float
``alpha`` attribute; unlike standard ONNX LeakyRelu, it supplies no default
attribute. Dynamic runtime shapes, scalars and zero-element tensors are
accepted up to rank 16. No static shape-inference callback is registered yet.

The descriptor requests C API 18 and uses status-returning ``CreateKernelV2``
and ``KernelComputeV2`` callbacks. ``KernelDestroy`` releases each retained
native kernel. Registration uses the supplied ``OrtApiBase``, never imports
``OrtGetApiBase``, and rejects hosts that cannot supply API 18.

The reverse bridge compiles the existing onnx-light ``kernel_leakyrelu.cc``
with its core dependency, not the full kernel collection. It owns the retained
``LeakyReluKernel`` adapter; the custom-op library owns ABI callbacks and
translates native exceptions to host statuses. The native kernel
is constructed once per ORT kernel instance. Each invocation borrows the
current input and ORT-allocated output directly, including preallocated outputs
provided through IO binding, without allocating or copying tensor payloads.
Borrowed views and context pointers are never retained across calls.

Before borrowing, the adapter validates CPU memory, float32 type, rank,
nonnegative dimensions, checked element/byte counts, and non-null, float-aligned
nonempty buffers. Input and output buffers must not overlap, including when
the caller supplies them through IO binding. It relies on the ORT host's valid
``OrtValue`` contract for actual allocation capacity and output shape/device;
API 18 does not expose
an independent buffer capacity. Native exceptions are translated to statuses
only at C ABI boundaries. ORT statuses are propagated unchanged.

Registration retains a mutex-protected domain and descriptor per supplied API
table, shared across sessions until library unload. Kernel attributes are
per-instance, not global. As required by the custom-op ABI, the application
must keep the library and its ORT host loaded until all consuming session
options, sessions and kernels have been destroyed.

Dependency boundary
-------------------

The dependency direction is this repository -> onnx-light. Keep only generic
kernel factories, executable-partition interfaces, and ownership/allocation
mechanisms in onnx-light. Do not add ``Ort*`` types to its public interfaces.
Any missing generic hooks require separate onnx-light PRs.

For now builds follow onnx-light's ``main`` branch. The public ORT headers are
pinned independently to commit
``db5f5eaff7c32e09e90520374ac19b78d89094ae``. Pinning headers does not claim
support for every function in their ABI.

Implementation order
--------------------

The host workstream starts with ABI tables and ownership-aware adapters,
then CPU custom ops, kernel-based EPs, and finally compiling EPs. It must
never fall back to an installed ORT runtime.

The reverse workstream starts with a CPU custom-op library exporting
``RegisterCustomOps``, followed by a kernel-based plugin EP. A compiling EP
is a later milestone. Use the real host's supplied API table and manual
initialization rather than importing ``OrtGetApiBase`` at static initialization.
The host shim is not a prerequisite for the reverse target.

In both directions retain kernel state across calls, validate types, shapes,
devices and buffer extents, and release plugin-owned resources before unloading
libraries. Native plugins are trusted executable code, not sandboxed inputs.
Persistent compilation caches, shared prepacked weights and accelerator
streams require explicit later contracts.
