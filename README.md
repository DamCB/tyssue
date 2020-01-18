# tyssue : An epithelium simulation library

![A nice banner](doc/illus/banner.png)

<hr/>

[![Build Status](https://travis-ci.org/DamCB/tyssue.svg?branch=master)](https://travis-ci.org/DamCB/tyssue)

[![Coverage Status](https://coveralls.io/repos/DamCB/tyssue/badge.svg)](https://coveralls.io/r/DamCB/tyssue)

[![Doc Status](https://readthedocs.org/projects/tyssue/badge/?version=latest)](http://tyssue.readthedocs.io/en/latest/
)

[![DOI](https://zenodo.org/badge/32533164.svg)](https://zenodo.org/badge/latestdoi/32533164) [![Join the chat at https://gitter.im/DamCB/tyssue](https://badges.gitter.im/DamCB/tyssue.svg)](https://gitter.im/DamCB/tyssue?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)


The `tyssue` library seeks to provide a unified interface to implement
bio-mechanical models of living tissues.
It's main focus is on **vertex based epithelium models**.

## Overview

### What kind of Models does it implement?

The first model implemented is the one described in
Monier et al. [monier2015apico]. It is an example of a vertex model,
where the interactions are only evaluated on the apical surface sheet
of the epithelium. The second class of models is still at an
stage. They implement a description of the tissue's rheology, within a
dissipation function formalism.

![The two models considered](doc/illus/two_models.png)

### General Structure of the modeling API

#### Design principles

> [APIs not apps](https://opensource.com/education/15/9/apis-not-apps)

Each biological question, be it in morphogenesis or cancer studies is
unique, and requires tweeking of the models developed by the
physicists. Most of the modelling softwares follow an architecture
based on a core C++ engine with a combinaison of markup or scripting
capacities to run specific simulation.

In `tyssue`, we rather try to expose an API that simplifies the
building of tissue models and running simulations, while keeping the
possibilities as open as possible.

> Separate structure, geometry and models

We seek to have a design as modular as possible, to allow the same
epithlium mesh to be fed to different physical models.

> Accessible, easy to use data structures

The core of the tyssue library rests on two structures: a set of
`pandas DataFrame` holding the tissue geometry and associated data,
and nested dictionnaries holding the model parameters, variables and
default values.

![Tyssue data structure](doc/illus/tyssue_data_management.png)

The API thus defines an `Epithelium` class. An instance of this class
is a container for the datasets and the specifications, and implements
methods to manipulate indexing of the dataframes to ease calculations.

The mesh structure is heavily inspired by
[CGAL Linear Cell Complexes](http://doc.cgal.org/latest/Linear_cell_complex/index.html),
most importantly, in the case of a 2D vertex sheet for example, each
junction edge between the cells is "splitted" between two oriented **half
edges**.


#### Creating an Epithelium

```python
## Core object
from tyssue import Sheet
## Simple 2D geometry
from tyssue import PlanarGeometry
## Visualisation (matplotlib based)
from tyssue.draw import sheet_view

sheet = Sheet.planar_sheet_2d('basic2D', nx=6, ny=7,
                              distx=1, disty=1)
PlanarGeometry.update_all(sheet)
sheet.sanitize()
fig, ax = sheet_view(sheet)
```

### Features

* Easy data manipulation.
* Multiple geometries (Sheets in 2D and 3D, monolayers, bulk).
* Easy to extend.
* 2D (matplotlib) and 3D (ipyvolume) customisable visualisation.
* Easy quasistatic model definition.
* Self collision detection. **new in 0.3**


### Documentation

* The documentation is browsable online [here](http://tyssue.readthedocs.io/en/latest/)
* Introduction notebooks are available [here](doc/notebooks).

### Mailing list:

tyssue@framaliste.org - https://framalistes.org/sympa/info/tyssue

Subscribe ➙ https://framalistes.org/sympa/subscribe/tyssue
Unsubscribe ➙ https://framalistes.org/sympa/sigrequest/tyssue


### Authors

* Bertrand Caré - @bcare
* Cyprien Gay - @cypriengay
* Guillaume Gay (maintainer) - @glyg
* Hadrien Mary - @hadim
* François Molino
* Magali Suzanne
* Sophie Theis - @sophietheis

## Dependencies

As all the dependencies are already completely supported in
python 3.x, **we won't be maintaining a python 2.x version**, because
it's time to move on...

### Core

- CGAL > 4.7
- Python >= 3.6
- numpy
- scipy
- matplotlib
- pandas
- pytables
- jupyter
- notebook
- quantities
- ipywidgets
- pythreejs
- ipyvolume
- vispy

### Tests

- pytest
- coverage
- pytest-cov

## Install

You can install the library with the conda package manager


```bash
conda install -c conda-forge tyssue
```


### Through PyPi

You can also install tyssue from PyPi, this is a CGAL-less version (pure python), lacking some features:

`python -m pip install --user --upgrade tyssue`

See [INSTALL.md](INSTALL.md) for a step by step install, including the necessary python environment.


## Licence

Since version 0.3, this project is distributed under the terms of the [General Public Licence](https://www.gnu.org/licenses/gpl.html).


Versions 2.4 and earlier were distributed under the [Mozilla Public Licence](https://www.mozilla.org/en-US/MPL/2.0/).

If GPL licencing is too restrictive for your intended usage, please contact the maintainer.

## Bibliography

* There is a [Bibtex file here](bibliography/tyssue.bib) with collected relevant publications.

The tyssue library stemed from a refactoring of the `leg-joint` code used in [monier2015apico].


[monier2015apico]: Monier, B. et al. Apico-basal forces exerted by
  apoptotic cells drive epithelium folding. Nature 518, 245–248 (2015).

[Tamulonis2013]: Tamulonis, C. Cell-based models. (Universiteit ven Amsterdam, 2013). doi:10.1177/1745691612459060.

[Tlili2013]: Tlili,S. et al. Mechanical formalism for tissue dynamics. 6, 23 (2013).
