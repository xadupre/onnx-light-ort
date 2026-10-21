Library components
==================

Each component has its own source directory and README. Implemented targets
own their CMake definitions and native tests; integration tests additionally
exercise the real ORT host. Planned components are identified explicitly,
without placeholder tests that would imply an implementation exists.

.. toctree::
   :maxdepth: 1

   custom_ops
   custom_ep
   bridge_light_to_ort
   bridge_ort_to_light
