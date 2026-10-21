ORT to onnx-light bridge
========================

``onnx_light_ort/bridge_ort_to_light/`` defines the interface target
``onnx_light_ort::bridge_ort_to_light``. The runtime implementation is planned:
it will load independent ORT-compatible custom-op and EP libraries into
onnx-light, exposing audited API tables and ownership-aware adapters.
It must build, test and run without installing, loading or linking ORT.

This direction is independent of :doc:`bridge_light_to_ort`. In particular,
the minimal host in ``tests/ort_test_host.h`` is explicit unit-test
infrastructure for the reverse direction, not this production bridge.

Current dependency tests
------------------------

.. code-block:: bash

   cmake -S . -B build -DONNX_LIGHT_ORT_BUILD_HOST=ON \
     -DONNX_LIGHT_ORT_BUILD_PLUGIN=OFF -DONNX_LIGHT_ORT_TEST_WITH_ORT=OFF
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build -L bridge_ort_to_light --output-on-failure

``tests/test_dependency_contract.cc`` lives in this component and checks
custom-domain model serialization and a real onnx-light custom kernel across
vector/matrix/scalar/empty shapes, including input preservation. It is labelled
``dependency``, not a unit test of a loader that does not exist yet.

The first implementation must add API negotiation, opaque-value ownership,
registration failure, loader teardown and independent fixture-plugin tests
alongside the actual bridge code. See :doc:`../next_steps` for the API inventory
and its no-ORT-runtime acceptance criteria.
