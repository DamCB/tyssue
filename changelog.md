
# What's new in 0.3

## Solvers

The `solvers.quasistatic.QSSolver` class provides a refactored solver that includes automatic Type 1, Type 3 and collision detection solving after each function evaluation. Use it with:

```
solver = QSSolver(with_t1=True, with_t3=True, with_t3=True)
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
