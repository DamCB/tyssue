.. Tyssue documentation master file, created by
   sphinx-quickstart on Mon Aug 22 16:57:50 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Tyssue's documentation!
==================================



The ``tyssue`` simulation library
=================================

The ``tyssue`` library seeks to provide a unified interface to implement
bio-mechanical models of living tissues, with a focus on **epithelium**
modeling. It's main focus is on **vertex** based models.

1. Overview
-----------

What kind of Models does it implement?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first model implemented is the one described in Monier et al.
[monier2015apico]. It is an example of a vertex model, where the
interactions are only evaluated on the apical surface sheet of the
epithelium. The second class of models are still at an stage. They
implement a description of the tissue's rheology, within a dissipation
function formalism.

.. figure:: illus/two_models.png
   :alt: The two models considered

   The two models considered

The mesh structure is heavily inspired by `CGAL Linear Cell
Complexes <http://doc.cgal.org/latest/Linear_cell_complex/index.html>`__,
most importantly, in the case of a 2D vertex sheet for example, each
junction edge between the cells is "splitted" between two oriented
**half edges**.

General Structure of the modeling API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core of the tyssue library rests on two structures: a set of
``pandas DataFrame`` holding the tissue geometry and associated data,
and a nested dictionnary holding the model parameters, variables and
default values.

.. figure:: illus/tyssue_data_management.png
   :alt: Tyssue data structure

   Tyssue data structure

The API thus defines an ``Epithelium`` class. An instance of this class
is a container for the datasets and the specifications, and implements
methods to manipulate indexing of the dataframes to ease calculations.

Contents:

.. toctree::
   :maxdepth: 2

   Basic creation and index manipulation <notebooks/Basic_Usage>
   Visualisation <notebooks/Visualization>
   Geometries <notebooks/Geometries_showcase>
   Energy minimization <notebooks/Energy_minimization>
   Cell Division <notebooks/Cell_Division>
   Type 1 transition <notebooks/Type_1_transition>
   Apoptosis <notebooks/Apoptosis>
   API

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
