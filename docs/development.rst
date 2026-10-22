Development
===========

Build against onnx-light main
-----------------------------

The default CMake configuration fetches onnx-light's ``main`` branch.
Python bindings, upstream tests, benchmarks and built-in operator kernels
are disabled in this bootstrap. Only the core dependency needed by the
native integration test is built; unused upstream targets are excluded
from the default build.

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build --output-on-failure

On Windows or other multi-configuration generators, also pass
``--config Release`` to the build and ``-C Release`` to CTest.
On macOS use ``sysctl -n hw.logicalcpu`` in place of ``nproc``.
Keep one build tree; reconfigure it rather than maintaining stale parallel
compiled copies.

Build either target independently:

.. code-block:: bash

   cmake -S . -B build -DONNX_LIGHT_ORT_BUILD_HOST=ON \
     -DONNX_LIGHT_ORT_BUILD_PLUGIN=OFF
   cmake --build build --parallel "$(nproc)"
   ctest --test-dir build --output-on-failure

Reverse the two options for a plugin-only build; both are enabled by default.
``ONNX_LIGHT_ORT_BUILD_TESTS=OFF`` disables native integration tests.

To use an existing checkout explicitly:

.. code-block:: bash

   git -C /path/to/onnx-light fetch origin main
   cmake -S . -B build \
     -DFETCHCONTENT_SOURCE_DIR_ONNX_LIGHT=/path/to/onnx-light

The override uses the checkout exactly as supplied and does not switch
branches. Ensure it contains the desired main revision; CMake logs its HEAD.
CI explicitly checks out upstream ``main`` and passes this override so its
resolved commit appears in the job logs. Fresh CI runs and the daily schedule
exercise new upstream main changes without a release.

Documentation and style
-----------------------

.. code-block:: bash

   python -m pip install -r requirements/style.txt -r docs/requirements.txt
   black . && ruff check .
   clang-format -i tests/test_dependency_contract.cc
   clang-format --dry-run --Werror tests/test_dependency_contract.cc
   python -m sphinx -b html -W --keep-going docs build/docs

The build and style tools ignore downloaded dependencies and generated output.
Tests use the real onnx-light core, without mocks or an installed Python
package. Nothing is installed into site-packages by the native build.

CI and documentation publishing
-------------------------------

Native CI covers Linux, Windows and macOS, including host-only, plugin-only
and combined configurations. It uses a compiler cache, cancels superseded
branch runs, and builds against upstream main. A Linux check rejects an ORT
shared-library dependency in the native test executable.

Fuzzing onnx-light
------------------

The optional fuzz targets are provided by the fetched onnx-light dependency
and require Clang with libFuzzer support:

.. code-block:: bash

   CC=clang CXX=clang++ cmake -S . -B build-fuzz \
     -DONNX_LIGHT_ORT_BUILD_FUZZERS=ON \
     -DONNX_LIGHT_ORT_BUILD_TESTS=OFF
   cmake --build build-fuzz --parallel "$(nproc)"
   ./build-fuzz/_deps/onnx_light-build/fuzz_checker -runs=2000

Fuzzing is disabled by default. The ``fuzz`` GitHub Actions workflow builds
the available upstream harnesses and runs short smoke campaigns; it does not
replace longer coverage-guided fuzzing.

Documentation builds on pull requests, main pushes and manual dispatch.
Only successful main-branch documentation builds deploy to GitHub Pages.
Workflow tokens are read-only by default; only the documentation deployment
job receives Pages and OIDC write permissions.
