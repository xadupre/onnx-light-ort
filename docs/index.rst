onnx-light-ort
==============

Interoperability between onnx-light and ONNX Runtime custom operators and
execution providers, in both directions.

**Status:** the reverse-direction shared library runs onnx-light's native
float32 CPU ``LeakyRelu`` inside real ORT. The standalone compatibility host
and execution providers remain planned.

.. toctree::
   :maxdepth: 2

   architecture
   components/index
   next_steps
   development
