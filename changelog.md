
# What's new in 0.5

## Major rewrite of the rearangements

We now allow for rosettes to form, and split type1 transition in two steps: merging of edges bellow the critical length and spliting more than rank 3 vertices (or more than rank 4 in 3D). The splitting frequency is governed by two settings `p_4` and `p5p`.This follows Finegan et al 2019. See  doc/notebooks/Rosettes.ipynb for a demo.

A look a the diffs in `sheet_topology` and `bulk_topology` should convince the reader that this should result in a major increase in stability.

Automated reconnection is now treated as an event (treated by an `EventManager` instance), see `tyssue.behavior.base_events.reconnect`.

In EulerSolver, this avoids raising `TopologyChangeError` at least in explicit Euler. Topology changes in IVPSolver are not supported for now.


## Viscous solver

- added a `bounds` attribute to EulerSolver. This simply clips the displacement to avoid runaway conditions.


## Core and topology

- A new `update_rank` method allows to compute the rank of a vertex (as the number of _flat_ edges connected to it). This required to move the `connectivity` module to utils to avoid circular imports.

- We explicitly allow two sided faces to be created by `collapse_edge` or `remove_face`, they are directly eliminated.


# What's new in 0.4

##  Time dependant solvers

- Added two time-dependant solvers `solvers.viscous.EulerSolver` and `solvers.viscous.IVPSolver` -- see [the demo](doc/notebooks/SimpleTimeDependent.ipynb)

- Added a connectivity matrices module

- removed layer type 1 transition (maybe a bit harshly)

- rewrote effectors specification with default values

- added a `merge_border` function to put single edges at a 2D sheet border, and a `trim_borders` option to the `sheet.remove()` and `sheet.sanitize()` methods

## Pipes

- collision detection should now be an optional dependency
- `doc/notebooks` is now synched with the [tyssue-demo](https://github.com/damcb.tyssue-demo) repo
- new `make nbtest` uses nbval to run the notebooks under pytest
(provinding some kind of integration tests)



# What's new in 0.3.1

- Collision detection also works for the outer layers of bulk tissues, i.e. collisions of the apical or basal surfaces are avoided for a monolayer.

- Added `get_neighbors` and `get_neighborhood` method to `Epithelium` to allow local patch queries in bulk epithelia (was initially only possible for 2D sheets).


# What's new in 0.3

## Solvers

The `solvers.quasistatic.QSSolver` class provides a refactored solver that includes automatic Type 1, Type 3 and collision detection solving after each function evaluation. Use it with:

```
solver = QSSolver(with_t1=True, with_t3=True)
solver.find_energy_min(sheet, **minimize_kwargs)
```

The function signature is a bit different from the previous `sheet_vertex_solver.Solver` as key-word arguments are directly passed to scipy `minimize` function. You thus need to replace:

```python
solver_kw = {'minimize': {'method': 'L-BFGS-B',
                          'options': {'ftol': 1e-8,
                                      'gtol': 1e-8}}}
solver.find_energy_min(sheet, **solver_kw)
```

by:

```python
solver_kw = {'method': 'L-BFGS-B',
             'options': {'ftol': 1e-8,
                         'gtol': 1e-8}}}
solver.find_energy_min(sheet, **solver_kw)
```

to use the new solver.
Note that `sheet_vertex_solver.Solver` is still available.

## Behavior

###  Event management refactoring

We refactored event management with a keyword arguments only design to make passing complex parameter dictionnaries  easier.


Actions and events where added for monolayer objects.

There is now an option in the manager `append` methods kwargs to add unique event or not.

## Licence

We switched to GPL to be able to use CGAL without worrying. If this is
a problem to you, we can offer accomodations.

## Vizualisation

The use of the top level `draw.sheet_view` function is encouraged. It is now possible to specify visibility at the single face level with a `"visible"` column in the face DataFrame.


## Core

* Added a `History` class to handle time series of sheet movements

## Geometry

* Lumen volume calculation on a new geometry class (#110)
* Create a new segment vertex category : lateral in Monolayer
* adds `finally` statement to scale_unscale utils
* Change 'sagittal' key word by 'lateral' key word


## Dynamics

### New quasitatic solver class.

### New effectors

* Add LumenVolumeElasticity effector
* added SurfaceTension effector

## Bug fixes

* reset catched ValueError to Exception waiting for pandas to publish 0.24
* Better opposite management and validation for Sheet, closes #72
* Correction of color face (#85)
* fixes reset_specs warning formatting bug
* Correction of segment category for new faces create in IH transition

## Misc

The codebase now uses [black](https://github.com/ambv/black) to format all the code base.

## Pruning

* removed old isotropic model
* removes multisheet (#105)
