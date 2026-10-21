onnx-light to ORT bridge
========================

``onnx_light_ort/bridge_light_to_ort/`` builds the static target
``onnx_light_ort::bridge_light_to_ort``. It adapts native onnx-light execution
to an ORT host's current tensors and kernel context. Both :doc:`custom_ops`
and the future :doc:`custom_ep` can use it without depending on each other.

The current ``LeakyReluKernel`` retains one native kernel and its alpha
attribute. Each ``Compute`` call validates CPU memory, float32 type, rank,
dimensions, checked element/byte counts, non-null nonempty buffers, alignment
and non-overlap before borrowing payloads. It owns neither the input/output
storage nor the ORT context and retains none of them between calls.
Native exceptions propagate to the consuming library's C ABI boundary.
Actual allocation capacity remains part of the real host's valid tensor
contract; see :doc:`../architecture`.

Unit tests
----------

.. code-block:: bash

   ctest --test-dir build -L bridge_light_to_ort --output-on-failure
   ctest --test-dir build -R bridge_light_to_ort_unit --output-on-failure

The component's ``tests/test_leaky_relu_kernel.cc`` exercises the real adapter
and numerical kernel using a minimal explicit C API test host. It covers
repeated vector/matrix/scalar/empty/rank-16 execution, input preservation,
missing inputs/outputs, device/type/rank rejection, negative dimensions,
element and byte-count overflow, null or overlapping buffers, unchanged
status propagation, metadata cleanup and recovery after errors.

These native tests need no ORT installation or mocking framework. The
fixture can supply invalid metadata that real ORT usually rejects before
dispatch. The separately enabled real-ORT suite also checks caller-provided
aligned/unaligned buffers and direct IO-bound outputs.
