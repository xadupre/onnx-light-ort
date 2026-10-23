onnx-light-ort
==============

.. image:: https://github.com/xadupre/onnx-light-ort/actions/workflows/docs.yml/badge.svg?branch=main
   :target: https://github.com/xadupre/onnx-light-ort/actions/workflows/docs.yml
   :alt: Documentation build status

.. image:: https://codecov.io/gh/xadupre/onnx-light-ort/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/xadupre/onnx-light-ort
   :alt: Native C++ code coverage

Native interoperability between onnx-light and ONNX Runtime, in both directions:

* Run onnx-light kernels as custom operators in ONNX Runtime.
* Run audited ORT custom-op libraries in standalone onnx-light, without loading
  or linking ONNX Runtime.

Custom operators are implemented; execution providers are planned.
The standalone host supports a limited single-node CPU custom-op profile,
not general ONNX graphs.

.. image:: docs/_static/projects.svg
   :target: https://xadupre.github.io/onnx-light-ort/
   :alt: Four bridges inside onnx-light-ort: custom operators and execution providers in both directions between onnx-light and ONNX Runtime. Custom operators are implemented; execution providers are planned.
   :align: center
   :width: 100%

Build and test
--------------

Requires Git, CMake 3.28 or later, a C++20 compiler and network access for the
first configuration. The native build fetches onnx-light's main branch, pinned
onnx-light-cpu computation sources and ORT's public C headers; it does not
require Python packages or an ONNX Runtime installation.

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build --output-on-failure

On multi-configuration generators, add ``--config Release`` to the build
command and ``-C Release`` to CTest. On macOS, replace ``nproc`` with
``sysctl -n hw.logicalcpu``.

See the `development guide
<https://xadupre.github.io/onnx-light-ort/development.html>`_ for local source
overrides, Python bindings, real-ORT integration tests and release packaging.

Documentation and examples
--------------------------

* `Documentation <https://xadupre.github.io/onnx-light-ort/>`_: project overview
  and component guides.
* `Custom operators
  <https://xadupre.github.io/onnx-light-ort/components/bridge_ops_light_called_by_ort.html>`_:
  supported kernels, Python/C++ usage and benchmark methodology.
* `Executable examples
  <https://xadupre.github.io/onnx-light-ort/auto_examples/index.html>`_:
  both custom-op directions and the ORT-relative Log benchmark.
* `Architecture <https://xadupre.github.io/onnx-light-ort/architecture.html>`_:
  bridge responsibilities and runtime boundaries.
* `GitHub Releases <https://github.com/xadupre/onnx-light-ort/releases>`_:
  Python wheels and native C++ archives.

Python models use ``onnx_light.onnx``, not the ``onnx`` Python package.

License
-------

`Apache-2.0 <LICENSE>`_. See `THIRD_PARTY_NOTICES.rst <THIRD_PARTY_NOTICES.rst>`_
for upstream license notices.
