onnxruntime custom operators called by onnx-light
=================================================

Run trusted ORT-compatible custom-op libraries in the standalone onnx-light
runtime, without installing or loading ONNX Runtime. This project is implemented.

Design
------

``onnx_light_ort::bridge_ops_ort_called_by_light`` is a shared library depending only
on ``common_ort_to_light`` and the platform loader. It does not install,
load or link ORT. ``onnx_light_ort.CustomOpSessionOrtCalledByLight`` accesses its small C ABI
through ``ctypes`` from the source checkout. Its model objects come from
``onnx_light.onnx`` bindings built in the same tree, not the ``onnx`` package;
no project artifact is installed into site-packages.
Execution uses a real onnx-light ``RuntimeSession`` and context-local
``KernelBase`` factory, not a substitute Python implementation.

C++ consumers link only ``onnx_light_ort::bridge_ops_ort_called_by_light`` and
include ``onnx_light_ort/bridge_ops_ort_called_by_light/host.h``. The target
publishes the repository and ORT API include paths; its implementation
libraries remain private. A consumer test builds without the test helper's
additional include paths to verify this contract.

Requirements
------------

Build with a C++20 compiler, CMake 3.28 or newer and
``ONNX_LIGHT_ORT_BUILD_HOST=ON``. CMake supplies onnx-light and pinned ORT
headers; no ORT runtime is needed. Use only explicitly selected, trusted native
plugins matching the audited API-18 profile below.
Python callers need NumPy, ml-dtypes and matching ``onnx_light.onnx`` bindings
from ``build/python``. See :doc:`../development` for source-build instructions.

Initial audited profile
------------------------

This is not a general ORT compatibility claim. Plugins are trusted executable
code, loaded from this package's library or an explicit caller-supplied path
resolved before loading. Models never select a library.
They must be audited against this API subset; unsupported function-table
slots are null and must not be called. Arbitrary plugin binaries are not safe
to probe for compatibility by executing them.

The host accepts only API-18 descriptors, one required CPU input and output of
matching float16, bfloat16, float32 or float64 type, a custom-domain opset import,
and status-returning ``CreateKernelV2``
and ``KernelComputeV2`` callbacks with ``KernelDestroy``. The initial model
must contain exactly one custom node, with distinct graph input and output,
no initializers and no local functions. Model data is limited to 16 MiB and
each runtime tensor to 64 MiB, with rank at most 16 and checked nonnegative
dimension products. Scalars and empty tensors are supported.
The Python wrapper checks serialized model size before parsing bytes and input
size and rank before copying arrays. These checks raise ``ValueError``;
native validation remains in place for C callers.

Registration rejects missing descriptors, unsupported devices or types,
duplicate registrations for the same domain/name/type, incompatible opsets,
optional/variadic arguments, alias callbacks and shape-inference callbacks.
Concrete type overloads are selected by the model's tensor type; generic
``UNDEFINED`` type descriptors are not supported. Legacy callbacks, EPs,
logging/configuration, general allocator APIs and the remaining roadmap
functions are not implemented. Output shapes are determined at execution.

Kernels are initialized lazily once per session node and retained across
changing input shapes. Every call reads the current runtime context.
The Python wrapper serializes runs and close operations, copies inputs to
protect caller-owned arrays from the API's mutable data accessor, and returns
independent NumPy output copies. Output buffers are host-owned. This initial
Python interface does not claim zero-copy execution.

``LightOrtRunTyped`` accepts a raw input pointer, its ONNX element type and
element count, and returns raw output storage of the same type. It checks
the model type, element alignment and dtype-dependent byte limit before reading
the input. The original ``LightOrtRun`` C ABI remains available for float32 and
rejects models of another type. The Python wrapper uses the typed entry point,
accepts NumPy float16/float32/float64 and ``ml_dtypes.bfloat16`` arrays, and rejects
arrays whose dtype differs from the model instead of converting them.

``close()`` and the context-manager protocol release kernels before unloading
the plugin. A finalizer also releases abandoned sessions. Registration and
compute errors propagate as Python exceptions; there is no ORT fallback.

C ABI contract
--------------

There are two C ABI boundaries: the ORT-compatible interface implemented for
plugins, and the small session interface exported to C++ and Python callers.
Neither requires an ORT runtime.

The host loads the plugin's exported ``RegisterCustomOps`` with this signature:

.. code-block:: cpp

   extern "C" OrtStatus *ORT_API_CALL
   RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api_base);

It supplies its own ``OrtApiBase``: ``GetApi`` returns the compatibility table
only for version 18, and ``GetVersionString`` identifies the standalone host.
``CreateCustomOpDomain``, ``CustomOpDomain_Add`` and ``AddCustomOpDomain`` record
the plugin's domains and descriptors. The plugin retains their ownership;
they must outlive the sessions that reference them. The supported function-table
entries are listed in `Shared support for hosting ORT-compatible plugins`_
below; all other slots are null, not fallback implementations.

The host validates each ``OrtCustomOp`` descriptor against the profile above,
selects it by domain, name, opset and tensor type, then calls ``CreateKernelV2``
once on first execution. It supplies ``KernelInfoGetAttribute_float`` for
construction attributes. On each run, ``KernelComputeV2`` receives an adapted
``OrtKernelContext``; input, output, type/shape and data-access functions map to
onnx-light CPU tensors. Metadata objects require
``ReleaseTensorTypeAndShapeInfo``; values and tensor buffers are borrowed and
must not be freed by the plugin. ``KernelDestroy`` releases retained state
before the plugin library unloads. Callbacks must return ``nullptr`` on success
or an owned ``OrtStatus *`` on failure, never let C++ exceptions cross the ABI.
The host reads errors through ``GetErrorMessage`` and releases statuses through
``ReleaseStatus``.

For application callers, ``host.h`` declares the separate exported interface:

* ``LightOrtCreateSession`` takes a trusted library path and serialized ONNX
  bytes, loads the plugin, parses the model and returns an opaque session.
* ``LightOrtRunTyped`` borrows element-aligned input storage for the call and
  returns session-owned output data and shape pointers. Copy them before the
  next run or session release. ``LightOrtRun`` is the float32-only convenience
  entry point. Serialize calls and release operations on a given session.
* ``LightOrtReleaseSession`` destroys the session, its kernels and then the
  library. It must not race a run.
* Failures from creation or execution return a non-null status. Read its
  borrowed message with ``LightOrtGetErrorMessage``, then free the status with
  ``LightOrtReleaseStatus``. Success returns ``nullptr``.

Tests and example
-----------------

The independent ``examples/custom_ops/ort_custom_ops.cc`` implements
``ort_custom_ops::LeakyRelu`` using only pinned ORT headers and its own numerical
code, without linking onnx-light. This independent example remains float32-only.
Native tests execute it under the standalone host, including
scalar/empty inputs and API negotiation failure. Python tests check changing
shapes, read-only and noncontiguous inputs, retained kernel state, independent
sessions and outputs, close ordering, missing attributes, unsupported graphs,
versions, names and dtypes. They assert that neither ORT nor the ``onnx`` package
was imported.

Run ``ctest --test-dir build -L bridge_ops_ort_called_by_light --output-on-failure``.
Enable ``ONNX_LIGHT_ORT_TEST_PYTHON=ON`` for the Python cases.
When the plugin target is enabled too, an additional Python suite loads the
actual ``onnx_light::LeakyRelu`` and ``onnx_light_cpu::Log`` custom ops in all four
types through the same standalone host.
These share numerical cases with the real-ORT suite: SIMD tails, changing shapes,
scalars, empty tensors, near-one and subnormal inputs, non-finite values,
read-only inputs and recovery after type/rank errors. The host remains generic;
it does not link the outgoing bridge or import ORT to execute this kernel.
See :doc:`../auto_examples/plot_custom_ops_ort_called_by_light` for the executable example and
:doc:`../development` for build and environment setup.

Python snippet
--------------

After building the host and independent example plugin, set
``ONNX_LIGHT_ORT_INCOMING_CUSTOM_OP_LIBRARY`` to the plugin's absolute path.
``ONNX_LIGHT_ORT_HOST_LIBRARY`` optionally selects the host library explicitly.
This excerpt constructs a model and runs the trusted custom operator without ORT:

.. literalinclude:: ../examples/plot_custom_ops_ort_called_by_light.py
   :language: python
   :start-at: import os
   :end-at: print("ORT custom op called by onnx-light:", result)

C++ example
-----------

The C++ custom operator below is the independent plugin consumed by the Python
snippet. It implements registration, an API-18 ``OrtCustomOp`` descriptor and
the retained LeakyRelu kernel using only ORT headers, with no ORT or onnx-light
link dependency. Its two extra counter exports are used only by tests.

Build the plugin and host, then execute the native C++ host tests:

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
     -DONNX_LIGHT_ORT_BUILD_HOST=ON -DONNX_LIGHT_ORT_BUILD_EXAMPLES=ON
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build -L bridge_ops_ort_called_by_light --output-on-failure

The native test constructs the same custom-domain model, creates a session
through ``LightOrtCreateSession`` and executes it with ``LightOrtRun``.
For application integration, link
``onnx_light_ort::bridge_ops_ort_called_by_light`` and include its ``host.h``,
not the private compatibility-table implementation.

.. literalinclude:: ../../examples/custom_ops/ort_custom_ops.cc
   :language: cpp

.. include:: common_ort_to_light.inc
