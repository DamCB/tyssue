# tyssue : An epithelium simulation library

[![Build Status](https://travis-ci.org/DamCB/tyssue.svg?branch=master)](https://travis-ci.org/DamCB/tyssue)

[![Coverage Status](https://coveralls.io/repos/CellModels/tyssue/badge.svg)](https://coveralls.io/r/CellModels/tyssue)


[![Doc Status](https://readthedocs.org/projects/tyssue/badge/?version=latest)](https://readthedocs.org/projects/tyssue/badge/?version=latest)


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
## Core object
from tyssue.core.sheet import Sheet
## Simple 2D geometry
from tyssue.geometry.planar_geometry import PlanarGeometry
## Visualisation (matplotlib based)
from tyssue.draw.plt_draw import sheet_view

sheet = Sheet.planar_sheet_2d('basic2D', nx=6, ny=7,
                              distx=1, disty=1)
PlanarGeometry.update_all(sheet)
sheet.sanitize()
```

### Features

* Easy data manipulation
* Multiple geometries (Sheets in 2D and 3D, monolayers, bulk, cell
centered models...)
* Easy to extend
* 2D (matplotlib) and 3D (vispy) customisable visualisation

### Documentation

Several notebooks are available [here](doc/notebooks).


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
