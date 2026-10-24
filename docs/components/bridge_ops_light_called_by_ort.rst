onnx-light called by onnxruntime through a custom operator
==========================================================

Run native onnx-light ``LeakyRelu`` and onnx-light-cpu ``Log`` kernels inside
ONNX Runtime through custom-domain operators. This project is implemented.

Design
------

``onnx_light_ort::bridge_ops_light_called_by_ort`` is a static library implementing
the custom-op kernel lifecycle. ``CreateLeakyRelu`` reads the required float
alpha and constructs retained state; ``ComputeLeakyRelu`` adapts the current
invocation; ``DestroyLeakyRelu`` releases the state.
``CreateLog``, ``ComputeLog`` and ``DestroyLog`` provide the same lifecycle for
the optimized onnx-light-cpu Log kernel, without attributes.

``custom_ops`` owns registration, descriptors and domain lifetimes.
This bridge owns the ops-specific callbacks, while ``common_light_to_ort``
owns native execution, tensor validation and shared status conversion.
The bridge never depends on the EP bridge or incoming compatibility code.
All C ABI callbacks prevent native exceptions from escaping.

Requirements
------------

Build with a C++20 compiler and CMake 3.28 or newer, with
``ONNX_LIGHT_ORT_BUILD_PLUGIN=ON``. CMake supplies onnx-light, the standalone
onnx-light-cpu kernels and pinned ORT headers. The plugin does not link ORT,
but execution requires a real ORT host supporting API 18.
Python examples additionally need ``requirements/ort-tests.txt`` and the
matching ``onnx_light.onnx`` bindings from ``build/python``.
See :doc:`../development` for source-build instructions.

C ABI contract
--------------

The plugin implements the ORT custom-op C ABI from ``onnxruntime_c_api.h``,
not an execution-provider ABI. Its exported entry point is:

.. code-block:: cpp

   extern "C" OrtStatus *ORT_API_CALL
   RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api_base);

``RegisterCustomOps`` must have shared-library export visibility. It obtains
the host's function table through ``GetApi`` with version 18; it does not
resolve or link ORT functions itself. Registration uses ``CreateCustomOpDomain``,
``CustomOpDomain_Add`` and ``AddCustomOpDomain`` to attach the ``onnx_light`` and
``onnx_light_cpu`` domains to the supplied options.

Each zero-initialized ``OrtCustomOp`` descriptor sets ``version = 18`` and
implements the name, execution-provider, input/output count, concrete element
type, required-argument, CPU memory and start/end opset-version queries.
The descriptor wires three lifecycle callbacks:

* ``CreateKernelV2`` receives the host's ``OrtApi`` and ``OrtKernelInfo`` and
  returns retained kernel state through ``void **``. LeakyRelu reads ``alpha``
  with ``KernelInfoGetAttribute_float``; Log has no attributes.
* ``KernelComputeV2`` receives that state and the current ``OrtKernelContext``.
  It uses ``KernelContext_GetInput``, tensor type/shape/data queries and
  ``KernelContext_GetOutput`` to adapt borrowed CPU buffers to native kernels.
  It must not retain invocation-specific values or data pointers.
* ``KernelDestroy`` releases the retained state once the host is finished.

Registration, creation and computation return ``nullptr`` on success or an
``OrtStatus *`` on failure. Exceptions are translated through ``CreateStatus``;
the caller owns the returned status and releases it with ``ReleaseStatus``.
Tensor metadata obtained with ``GetTensorTypeAndShape`` is released with
``ReleaseTensorTypeAndShapeInfo``, but context-owned input/output values and
buffers must not be freed by the plugin. Descriptors and domains remain alive
for all referencing sessions; domains are cached per API table and released
with ``ReleaseCustomOpDomain`` when the plugin unloads. The host and its API
table must outlive that cleanup.

Tests
-----

.. code-block:: bash

   ctest --test-dir build -L bridge_ops_light_called_by_ort --output-on-failure

The native suite in ``tests/test_custom_ops_light_called_by_ort.cc`` verifies explicit zero alpha,
independent retained instances, exception translation and reuse after failure.
It uses the explicit minimal test host, not an installed ORT runtime.
The real-ORT integration suite also carries this label and checks the complete
entry-point/bridge/common chain.

.. include:: common_light_to_ort.inc

.. include:: custom_ops.inc
