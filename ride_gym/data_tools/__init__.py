"""Data-preparation tools for the ride_gym NYC / OSMnx scenarios.

This subpackage contains ONLY the *code* that downloads and preprocesses the
real-world assets the graph-based scenarios consume -- road-network builders
(OSMnx), the NYC FHVHV order preprocessor, taxi-zone centroid extraction, and
the train/val/test window splitter. It deliberately ships **no data**: the
cached graphs, distance matrices, zone-centroid CSVs and order parquets are
large and/or licence-encumbered, so users generate them locally by running
these tools (see each module's ``main()`` / ``python -m`` entry point).

By default every tool writes its outputs under ``./data/`` (relative to the
current working directory), NOT inside the installed package, so a pip-installed
read-only ``ride_gym`` still works and generated artefacts live in the user's
project.

Optional dependencies
---------------------
These tools need extras beyond the core ``ride_gym`` runtime:

* ``osmnx``, ``networkx`` -- road-network builders (``build_network``,
  ``nyc.build_nyc_network``);
* ``pandas``, ``pyarrow`` -- order preprocessing / splitting;
* ``geopandas`` -- taxi-zone centroid extraction.

Install them via the ``data`` extra: ``pip install ride_gym[data]``.
"""
