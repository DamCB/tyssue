# tyssue : An epithelium simulation library

[![Build Status](https://travis-ci.org/DamCB/tyssue.svg?branch=master)](https://travis-ci.org/DamCB/tyssue)

[![Coverage Status](https://coveralls.io/repos/CellModels/tyssue/badge.svg)](https://coveralls.io/r/CellModels/tyssue)

The `tyssue` library seeks to provide a unified interface to implement
bio-mechanical models of living tissues, with a focus on **epithelium** modeling.
It's main focus is on **vertex** based models.

## Overview

### What kind of Models does it implement?

The first model implemented is the one described in
Monier et al. [monier2015apico]. It is an example of a vertex model,
where the interactions are only evaluated on the apical surface sheet
of the epithelium. The second class of models are still at an
stage. They implement a description of the tissue's rheology, within a
dissipation function formalism.

![The two models considered](doc/illus/two_models.png)

The mesh structure is heavily inspired by
[CGAL Linear Cell Complexes](http://doc.cgal.org/latest/Linear_cell_complex/index.html),
most importantly, in the case of a 2D vertex sheet for example, each
junction edge between the cells is "splitted" between two oriented **half
edges**.

### General Structure of the modeling API

The core of the tyssue library rests on two structures: a set of
`pandas DataFrame` holding the tissue geometry and associated data,
and nested dictionnaries holding the model parameters, variables and default values.

![Tyssue data structure](doc/illus/tyssue_data_management.png)

The API thus defines an `Epithelium` class. An instance of this class
is a container for the datasets and the specifications, and implements
methods to manipulate indexing of the dataframes to ease calculations.

#### Creating an Epithelium

```python
from scipy.spatial import Voronoi
from tyssue.core.generation import hexa_grid2d, from_2d_voronoi
from tyssue.core.objects import Epithelium
from tyssue.config.geometry import planar_spec

from tyssue.core.generation import hexa_grid2d, from_2d_voronoi
grid = hexa_grid2d(nx=6, ny=4,
	               distx=1, disty=1)
datasets = from_2d_voronoi(Voronoi(grid))

eptm = Epithelium('2D_example', datasets,
                  specs=planar_spec(),
                  coords=['x', 'y'])
```

#### Datasets

Geometries and models are defined independently (or as independently
as possible) from the data. They are implemented as classes that are
not holding any data but rather define static and class methods that
act on the data hold by the epithelium objects.

```python
for key, df in eptm.datasets.items():
    print(key, df.shape)

>>> face (24, 6)
>>> edge (82, 7)
>>> vert (32, 3)
```

In 2D, we have 3 datasets, `'face'`, `'edge` and `'vert'`. They hold
the information for the cell faces (counfounded with the whole cell
here), edges and vertices, respectively.

Each of those can be accessed
as an attribute of the `Epithelium` object directly with a `_df`
suffix  to the element, e.g `eptm.face_df`.

#### Upcasting

Geometry are physics computations often require to access for example
the cell related data on each of the cell's edges. The `Epithelium`
class defines utilities to make this, i.e copying the values of a cell
associated data to each edges of the cell.


### Authors

* Cyprien Gay @cypriengay
* Guillaume Gay (maintainer) - @glyg
* Hadrien Mary (build wizard) - @hadim
* François Molino
* Magali Suzanne


## Dependencies

As all the dependencies are already completely supported in python 3.x, we won't be maintaining a
python 2.x version, because it's time to move on...

### Python

- Python >= 3.4
- numpy >= 1.8
- scipy >= 0.12
- pandas >= 0.13
- matplotlib >= 1.3
- vispy >= 0.4
- pandas >= 0.16


## Install

See [INSTALL.md](INSTALL.md)


## Similar softwares

* See [this publication](http://bioinformatics.oxfordjournals.org.gate1.inist.fr/content/32/2/219/F2.expansion.html) by Jereky's lab, softawre available [here](http://www.biocenter.helsinki.fi/salazar/software.html)


## Licence

This project is distributed under the terms of the [Modzilla Public Licence](https://www.mozilla.org/en-US/MPL/2.0/).


## Bibliography

* Here is a [Mendeley group](https://www.mendeley.com/groups/7132031/tyssue/) for the project's
  bibliography

* There is also a good old [Bibtex file here](bibliography/tyssue.bib)



[monier2015apico]: Monier, B. et al. Apico-basal forces exerted by
  apoptotic cells drive epithelium folding. Nature 518, 245–248 (2015).

[Tamulonis2013]: Tamulonis, C. Cell-based models. (Universiteit ven Amsterdam, 2013). doi:10.1177/1745691612459060.

[Tlili2013]: Tlili,S. et al. Mechanical formalism for tissue dynamics. 6, 23 (2013).

[1]: The fact that the LCC model uses the term `cell` as it's core
  concept is unfortunate. This will be hidden in the python API of the project.
