Examples gallery
================

These examples execute both custom-op directions using explicit library
paths. They do not implement or register an execution provider.

Build the native libraries first and set ``ONNX_LIGHT_ORT_PLUGIN_LIBRARY``,
``ONNX_LIGHT_ORT_HOST_LIBRARY`` and ``ONNX_LIGHT_ORT_INCOMING_CUSTOM_OP_LIBRARY``
to their absolute paths. Models are built and checked with ``onnx_light.onnx``,
not the ``onnx`` package. See :doc:`../development` for the commands and the
build-tree ``PYTHONPATH``.
