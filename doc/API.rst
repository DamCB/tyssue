
Core objects
============

The Epithelium object
---------------------

.. autoclass::   tyssue.Epithelium
   :members:
   :undoc-members:


Sheet object
------------

.. autoclass::   tyssue.Sheet
   :members:
   :undoc-members:


Monolayer object
----------------

.. autoclass::   tyssue.Monolayer
   :members:
   :undoc-members:


Drawing functions
=================

Matplotlib based
----------------

.. automodule::    tyssue.draw.plt_draw
   :members:
   :undoc-members:

Ipyvolume based
----------------

.. automodule::    tyssue.draw.ipv_draw
   :members:
   :undoc-members:


Geometry classes
================

Planar
------

.. autoclass:: tyssue.PlanarGeometry
   :members:
   :undoc-members:

Sheet (2D 1/2)
--------------

.. autoclass:: tyssue.SheetGeometry
   :members:
   :undoc-members:

.. autoclass:: tyssue.geometry.sheet_geometry.ClosedSheetGeometry
   :members:
   :undoc-members:

.. autoclass:: tyssue.geometry.sheet_geometry.EllipsoidSheetGeometry
   :members:
   :undoc-members:


Bulk (3D)
---------

.. autoclass:: tyssue.BulkGeometry
   :members:
   :undoc-members:

.. autoclass:: tyssue.MonolayerGeometry
   :members:
   :undoc-members:

.. autoclass:: tyssue.geometry.monolayer_geometry.ClosedMonolayerGeometry
   :members:
   :undoc-members:


Topology functions
==================

Base
----

.. automodule:: tyssue.topology.base_topology
   :members:
   :undoc-members:

Sheet
-----

.. automodule:: tyssue.topology.sheet_topology
   :members:
   :undoc-members:


Bulk and Monolayer
------------------

.. automodule:: tyssue.topology.bulk_topology
   :members:
   :undoc-members:

.. automodule:: tyssue.topology.monolayer_topology
   :members:
   :undoc-members:


Dynamic models definitions
==========================

Gradients
---------

.. automodule:: tyssue.dynamics.base_gradients
   :members:
   :undoc-members:

.. automodule:: tyssue.dynamics.planar_gradients
   :members:
   :undoc-members:

.. automodule:: tyssue.dynamics.sheet_gradients
   :members:
   :undoc-members:

.. automodule:: tyssue.dynamics.bulk_gradients
   :members:
   :undoc-members:


Effectors and Model factory
---------------------------

.. automodule:: tyssue.dynamics.factory
   :members:
   :undoc-members:

.. automodule:: tyssue.dynamics.effectors
   :members:
   :undoc-members:


Predefined models
-----------------

.. automodule:: tyssue.dynamics.planar_vertex_model
   :members:
   :undoc-members:

.. automodule:: tyssue.dynamics.sheet_vertex_model
   :members:
   :undoc-members:

.. automodule:: tyssue.dynamics.apoptosis_model
   :members:
   :undoc-members:

.. automodule:: tyssue.dynamics.bulk_model
   :members:
   :undoc-members:


Quasistatic solver
------------------

.. autoclass:: tyssue.solvers.quasistatic_solver.QSSolver
   :members:
   :undoc-members:




Epithelium generation utilities
===============================

.. automodule:: tyssue.generation
   :members:
   :undoc-members:

.. automodule:: tyssue.generation.shapes
   :members:
   :undoc-members:

.. automodule:: tyssue.generation.modifiers
   :members:
   :undoc-members:

.. automodule:: tyssue.generation.hexagonal_grids
   :members:
   :undoc-members:

.. automodule:: tyssue.generation.from_voronoi
   :members:
   :undoc-members:


Input/output
============

.. automodule:: tyssue.io.hdf5
   :members:
   :undoc-members:


.. automodule:: tyssue.io.csv
   :members:
   :undoc-members:

.. automodule:: tyssue.io.obj
   :members:
   :undoc-members:


Predefined datasets
-------------------

.. automodule:: tyssue.stores
   :members:
   :undoc-members:


Collision detection and correction
==================================

Detection
---------

.. automodule:: tyssue.collisions.intersection
   :members:
   :undoc-members:


Resolution
----------

.. automodule:: tyssue.collisions.solvers
   :members:
   :undoc-members:


Miscellanous utils
==================

.. automodule:: tyssue.utils.utils
   :members:
   :undoc-members:


Decorators
----------

.. automodule:: tyssue.utils.decorators
   :members:
   :undoc-members:
