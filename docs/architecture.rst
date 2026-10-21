Architecture
============

Two targets, one repository
---------------------------

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
       only public headers. Real-ORT integration tests will be opt-in.

At bootstrap these are interface targets only. The dependency test compiles
ORT's public C/EP declarations alongside onnx-light headers, links onnx-light's
core, round-trips a custom-domain model, and executes a native custom kernel
with changing shapes, including scalar and empty tensors. It does not prove
that an ORT plugin can already be loaded.

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
