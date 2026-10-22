onnx-light-ort
==============

.. image:: docs/_static/onnx-light-ort.svg
   :alt: onnx-light and ONNX Runtime interoperability
   :align: center
   :width: 250px

.. image:: https://github.com/xadupre/onnx-light-ort/actions/workflows/docs.yml/badge.svg?branch=main
   :target: https://github.com/xadupre/onnx-light-ort/actions/workflows/docs.yml
   :alt: Documentation build status

.. image:: https://codecov.io/gh/xadupre/onnx-light-ort/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/xadupre/onnx-light-ort
   :alt: Native C++ code coverage

Bidirectional interoperability between onnx-light and ONNX Runtime custom
operators and execution providers.

**Custom ops in both directions:** a shared library exposes onnx-light's native
CPU ``LeakyRelu`` kernel and onnx-light-cpu's optimized ``Log`` to real ONNX Runtime.
A standalone host executes audited
API-18 CPU custom-op libraries in onnx-light, initially for single-node models.
Both native operators expose float16, bfloat16, float32 and float64 in both
directions, preserving the input dtype. Execution providers are not implemented.

Four bridges and two common libraries
-------------------------------------

Sources live under ``onnx_light_ort/``, with a directory and CMake definition
per implemented component:

* ``custom_ops/``: the ORT-loadable custom-op shared library, exporting
  ``RegisterCustomOps``.
* ``custom_ep/``: the future ORT-loadable execution-provider library, with
  separate factory exports. It is not implemented and has no target yet.
* ``bridge_ops_light_called_by_ort/``: custom-op kernel construction, execution and
  destruction callbacks for running onnx-light inside ORT.
* ``bridge_ep_light_called_by_ort/``: future EP-specific adaptation in the same
  direction; currently an interface target.
* ``bridge_ops_ort_called_by_light/``: shared custom-op host and trusted explicit-path
  loading, with a source-checkout Python interface.
* ``bridge_ep_ort_called_by_light/``: future EP hosting inside onnx-light; currently
  an interface target.
* ``common_light_to_ort/``: shared native execution, buffer validation and
  status conversion for the two bridges targeting ORT.
* ``common_ort_to_light/``: static API-table and opaque-value support for the
  initial incoming custom-op profile.

``custom_ops`` links ``bridge_ops_light_called_by_ort``, which links
``common_light_to_ort``. The future EP uses its own bridge and the same common
library. Neither direction links the other direction's common library.
The existing ``::plugin`` and ``::host``
aliases remain available for custom ops and ``::common_ort_to_light``
respectively. No component links ONNX Runtime; reverse-direction integration
tests explicitly opt into a real ORT installation. The custom-op library uses
only its host-supplied API table, requesting C API version 18.

The first operator is ``onnx_light::LeakyRelu`` (custom-domain opset 1), with
one floating-point CPU input and output, and a required float ``alpha`` attribute.
It supports changing shapes, scalars and empty tensors up to rank 16.
The adapter borrows input and ORT-allocated output buffers without copying
their payloads and retains one native kernel per ORT kernel instance.

``onnx_light_cpu::Log`` (custom-domain opset 1) uses the standalone
``ml_cpu_compute`` library from onnx-light-cpu, with runtime SIMD dispatch.
The same plugin also runs through the standalone incoming bridge, without ORT.
It accepts one floating-point CPU tensor, preserves its shape and dtype, and has
no attributes. Each operator registers four concrete typed descriptors under
the same name, selected using the model's tensor type without implicit casts.
Both executable gallery examples demonstrate this kernel.
Earlier ``session.run`` measurements of the benchmark in
``docs/examples/plot_benchmark_log_custom_ops.py`` found float32 inference
3.09--3.42x faster than standard ORT Log for 32,768--1,048,576
elements on a Xeon Platinum 8480C with one thread. Standalone Python timings
include protective copies and do not imply the same speedup; see
``docs/components/custom_ops.rst`` for those results and methodology.
The current benchmark uses ORT I/O binding and plots speedups against built-in ORT Log.

Build and test
--------------

Version **0.2.0** adds automatic tag-triggered GitHub Releases with CPython
3.12, 3.13 and 3.14 wheels, plus native C++ archives. Targets match onnx-light:
Linux x86_64/aarch64, Windows AMD64/ARM64 and macOS universal2.
See ``docs/development.rst`` for release validation, artifact contents and
manual dry runs. No release is published until all platform builds succeed.

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
Plugin builds also fetch a pinned onnx-light-cpu revision, configuring only its
standalone computation library: no second onnx-light build or Python package.
See ``docs/development.rst`` for independent-target builds and source overrides.
Each component has its own ``README.md`` and a page under ``docs/components/``.
Native tests are selected with CTest labels ``custom_ops``,
``bridge_ops_light_called_by_ort``, ``common_light_to_ort`` and
``common_ort_to_light``; planned runtime components
are explicitly distinguished from implemented unit suites.

Using release wheels
--------------------

Wheels include both native libraries. ``get_library_path("custom_ops")`` returns
the ORT-loadable plugin and ``get_library_path("host")`` returns the standalone
host. Both return absolute paths and fail explicitly when a library is absent;
source checkouts continue using explicit build-tree paths.

.. code-block:: python

   from onnx_light_ort import CustomOpSession, get_library_path

   with CustomOpSession(
       model,
       library=get_library_path("custom_ops"),
       host_library=get_library_path("host"),
   ) as session:
       outputs = session.run(None, {"X": values})

Model construction and the standalone Python session require ``onnx_light.onnx``.
Obtain onnx-light from its GitHub release wheels or use a matching source build
through ``PYTHONPATH``; it is not currently published on PyPI. The optional
``models`` extra declares that dependency. Loading the plugin path for a real
ORT host does not import onnx-light; install ORT separately if using that host.
Never install the ``onnx`` Python package for these examples.

To test the custom-op library in a real ORT host:

.. code-block:: bash

   python -m pip install -r requirements/ort-tests.txt
   cmake -S . -B build -DONNX_LIGHT_ORT_TEST_WITH_ORT=ON
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build --output-on-failure

Runnable Python and C++ applications are under ``examples/custom_ops/``.
See ``docs/components/custom_ops.rst`` for their commands and CTest setup.
The executable Python gallery in ``docs/examples/`` demonstrates both custom-op
directions. See ``docs/development.rst`` for the standalone Python tests and
the gallery's explicit library-path configuration.
Python models use ``onnx_light.onnx`` from bindings built alongside the native
libraries in ``build/python``; the ``onnx`` Python package is not used.

Documentation
-------------

Documentation executes both Python gallery examples. First build the native
libraries and set their explicit paths as described in
``docs/development.rst`` under "Executable Python gallery".

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
