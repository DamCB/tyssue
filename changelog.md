
* WIP solver refactoring
* WIP constrained points repartition
* WIP energy min with collitions and T1


# What's new in 0.3

* passed black formatter


## Licence

We switched to GPL to be able to use CGAL without worrying. If this is
a problem to you, we can offer accomodations.

## Vizualisation


## Behavior

* event management refactoring
* Add monolayer actions
* Add option in kwargs to add unique event or not
* refactored `delamination` to work with the new event API

## Core

* Adds a History class to handle time series of sheet movements

## Geometry

* Lumen volume calculation on a new geometry class (#110)
* Create a new segment vertex category : lateral in Monolayer
* adds finally statement to scale_unscale utils
* Change 'sagittal' key word by 'lateral' key word

## Topology

* Correction of segment category for new faces create in IH transition


## Solvers

* Collisions (#102)


## Dynamics

* Add LumenVolumeElasticity effector
* added SurfaceTension effector

## Bug fixes

* reset catched ValueError to Exception waiting for pandas to publish 0.24
* Better opposite management and validation for Sheet, closes #72
* Correction of color face (#85)
* fixes reset_specs warning formatting bug

## Pruning

* removed old isotropic model
* removes multisheet (#105)
