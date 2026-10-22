onnx-light-ort
==============

.. image:: https://github.com/xadupre/onnx-light-ort/actions/workflows/docs.yml/badge.svg?branch=main
   :target: https://github.com/xadupre/onnx-light-ort/actions/workflows/docs.yml
   :alt: Documentation build status

.. image:: https://codecov.io/gh/xadupre/onnx-light-ort/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/xadupre/onnx-light-ort
   :alt: Native C++ code coverage

Interoperability between onnx-light and ONNX Runtime custom operators and
execution providers, in both directions.

**Status:** custom ops execute in both directions: native onnx-light
``LeakyRelu`` in real ORT, and audited API-18 CPU custom-op libraries in
onnx-light through a limited single-node float32 host. Execution providers
remain planned.

.. toctree::
   :maxdepth: 2

   architecture
   components/index
   auto_examples/index
   newt_steps/2026/2026-10-start
   development
