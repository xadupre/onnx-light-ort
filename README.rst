onnx-light-ort
==============

Bidirectional interoperability between onnx-light and ONNX Runtime custom
operators and execution providers.

**Bootstrap stage:** this repository provides the C++ build, dependency
integration tests, documentation and CI. Plugin loading, custom-op exports and
execution-provider implementations are not implemented yet.

Two independent directions
--------------------------

* ``onnx_light_ort::host`` prepares the target for hosting ORT-compatible
  custom kernels and plugin EPs in onnx-light, without an ORT runtime.
* ``onnx_light_ort::plugin`` prepares the target for exposing onnx-light
  kernels to an ORT host, first as custom ops and later as a plugin EP.

These are currently CMake interface targets, not working plugin libraries.
They share dependency declarations, not a compatibility API implementation.
Neither target links ONNX Runtime; future reverse-direction integration tests
will explicitly opt into a real ORT installation.

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

Documentation
-------------

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m sphinx -b html -W --keep-going docs build/docs

Documentation is deployed to
https://xadupre.github.io/onnx-light-ort/ after successful main-branch builds.
The design is described in ``docs/architecture.rst`` and the
`onnx-light roadmap
<https://github.com/xadupre/onnx-light/pull/5479>`_.

License
-------

Apache-2.0. Downloaded ORT headers retain their upstream MIT license notices;
see ``THIRD_PARTY_NOTICES.rst``.
