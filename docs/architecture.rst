Architecture
============

Two targets, one repository
---------------------------

See :doc:`next_steps` for the detailed implementation roadmap and the
inventory of host functions and plugin callbacks.

The compatibility host and the plugin for ORT are separate build targets.
They can eventually share type/shape conversion and buffer-contract helpers,
but never compatibility API tables or bootstrap exports.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Target
     - Direction
     - ORT runtime dependency
   * - ``onnx_light_ort::host``
     - Independent custom kernels and plugin EPs run in onnx-light.
     - None, including its build and tests.
   * - ``onnx_light_ort::plugin``
     - onnx-light kernels run inside an ORT host.
     - The consuming application supplies ORT; production builds need
       only public headers. Real-ORT integration tests are opt-in.

The host remains an interface target. The plugin is a shared library exporting
``RegisterCustomOps``. The dependency test compiles
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

The implementation compiles the existing onnx-light ``kernel_leakyrelu.cc``
with its core dependency, not the full kernel collection. The native kernel
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
