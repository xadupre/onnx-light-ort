onnx-light-ort
==============

Bidirectional interoperability between onnx-light and ONNX Runtime custom
operators and execution providers.

**First custom-op milestone:** a shared library exposes onnx-light's native
CPU ``LeakyRelu`` kernel to real ONNX Runtime. The standalone compatibility
host and execution providers are not implemented yet.

Two independent directions
--------------------------

* ``onnx_light_ort::host`` prepares the target for hosting ORT-compatible
  custom kernels and plugin EPs in onnx-light, without an ORT runtime.
* ``onnx_light_ort::plugin`` builds a shared library exposing onnx-light
  kernels to an ORT host as custom ops. Plugin EPs are a later milestone.

The host remains an interface target. Neither target links ONNX Runtime;
reverse-direction integration tests explicitly opt into a real ORT installation.
The plugin exports ``RegisterCustomOps`` and uses only its host-supplied API
table, requesting C API version 18.

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

To test the custom-op library in a real ORT host:

.. code-block:: bash

   python -m pip install -r requirements/ort-tests.txt
   cmake -S . -B build -DONNX_LIGHT_ORT_TEST_WITH_ORT=ON
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build --output-on-failure

Documentation
-------------

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m sphinx -b html -W --keep-going docs build/docs

Documentation is deployed to
https://xadupre.github.io/onnx-light-ort/ after successful main-branch builds.
The design is described in ``docs/architecture.rst`` and the implementation
roadmap in `docs/next_steps.rst <docs/next_steps.rst>`_.

License
-------

Apache-2.0. Downloaded ORT headers retain their upstream MIT license notices;
see ``THIRD_PARTY_NOTICES.rst``.
