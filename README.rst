onnx-light-ort
==============

.. image:: docs/_static/onnx-light-ort.svg
   :alt: onnx-light and ONNX Runtime interoperability
   :align: center
   :width: 250px

.. image:: https://github.com/xadupre/onnx-light-ort/actions/workflows/docs.yml/badge.svg?branch=main
   :target: https://github.com/xadupre/onnx-light-ort/actions/workflows/docs.yml
   :alt: Documentation build status

Bidirectional interoperability between onnx-light and ONNX Runtime custom
operators and execution providers.

**First custom-op milestone:** a shared library exposes onnx-light's native
CPU ``LeakyRelu`` kernel to real ONNX Runtime. The standalone compatibility
host and execution providers are not implemented yet.

Four bridges and two common libraries
-------------------------------------

Sources live under ``onnx_light_ort/``, with a directory and CMake definition
per implemented component:

* ``custom_ops/``: the ORT-loadable custom-op shared library, exporting
  ``RegisterCustomOps``.
* ``custom_ep/``: the future ORT-loadable execution-provider library, with
  separate factory exports. It is not implemented and has no target yet.
* ``bridge_ops_light_to_ort/``: custom-op kernel construction, execution and
  destruction callbacks for running onnx-light inside ORT.
* ``bridge_ep_light_to_ort/``: future EP-specific adaptation in the same
  direction; currently an interface target.
* ``bridge_ops_ort_to_light/``: future custom-op hosting inside onnx-light;
  currently an interface target.
* ``bridge_ep_ort_to_light/``: future EP hosting inside onnx-light; currently
  an interface target.
* ``common_light_to_ort/``: shared native execution, buffer validation and
  status conversion for the two bridges targeting ORT.
* ``common_ort_to_light/``: shared API-table, opaque-value and lifetime support
  for the two bridges targeting onnx-light; currently an interface target.

``custom_ops`` links ``bridge_ops_light_to_ort``, which links
``common_light_to_ort``. The future EP uses its own bridge and the same common
library. Neither direction links the other direction's common library.
The existing ``::plugin`` and ``::host``
aliases remain available for custom ops and ``::common_ort_to_light``
respectively. No component links ONNX Runtime; reverse-direction integration
tests explicitly opt into a real ORT installation. The custom-op library uses
only its host-supplied API table, requesting C API version 18.

The first operator is ``onnx_light::LeakyRelu`` (custom-domain opset 1), with
one float32 CPU input and output, and a required float ``alpha`` attribute.
It supports changing shapes, scalars and empty tensors up to rank 16.
The adapter borrows input and ORT-allocated output buffers without copying
their payloads and retains one native kernel per ORT kernel instance.

Build and test
--------------

Requires Git, CMake 3.28 or later, a C++20 compiler, and network access for
the first configuration. OpenSSL and OpenMP are optional upstream dependencies.
No Python package, ONNX Runtime installation or system-wide onnx-light
installation is needed for the C++ build.

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build --output-on-failure

The build fetches **onnx-light's main branch**, not a released package or a
version tag, and logs the resolved commit. ORT's public C headers are fetched
from an immutable commit and checked with SHA-256; no ORT runtime is fetched.
See ``docs/development.rst`` for independent-target builds and source overrides.
Each component has its own ``README.md`` and a page under ``docs/components/``.
Native tests are selected with CTest labels ``custom_ops``,
``bridge_ops_light_to_ort``, ``common_light_to_ort`` and
``common_ort_to_light``; planned runtime components
are explicitly distinguished from implemented unit suites.

To test the custom-op library in a real ORT host:

.. code-block:: bash

   python -m pip install -r requirements/ort-tests.txt
   cmake -S . -B build -DONNX_LIGHT_ORT_TEST_WITH_ORT=ON
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build --output-on-failure

Runnable Python and C++ applications are under ``examples/custom_ops/``.
See ``docs/components/custom_ops.rst`` for their commands and CTest setup.

Documentation
-------------

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m sphinx -b html -W --keep-going docs build/docs

Documentation is deployed to
https://xadupre.github.io/onnx-light-ort/ after successful main-branch builds.
The design is described in ``docs/architecture.rst`` and the implementation
roadmap in `docs/newt_steps/2026/2026-10-start.rst <docs/newt_steps/2026/2026-10-start.rst>`_.

License
-------

Apache-2.0. Downloaded ORT headers retain their upstream MIT license notices;
see ``THIRD_PARTY_NOTICES.rst``.
