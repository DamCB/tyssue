# tyssue : An epithelium simulation library


## About this fork

This repository is a fork of [DamCB/tyssue](https://github.com/DamCB/tyssue),
created by Guillaume Gay, Sophie Theis, Magali Suzanne and the tyssue contributors.
With the agreement of the original authors, tyssue is developed here by the
[vivarium-collective](https://github.com/vivarium-collective).

Releases are published on PyPI under the name `tyssue`.

![A nice banner](doc/illus/banner.png)

<hr/>

| Version | Downloads | Python |
| --- | --- | --- |
| [![PyPI version](https://img.shields.io/pypi/v/tyssue.svg)](https://pypi.org/project/tyssue/) | [![PyPI downloads](https://img.shields.io/pypi/dm/tyssue.svg)](https://pypi.org/project/tyssue/) | [![Python versions](https://img.shields.io/pypi/pyversions/tyssue.svg)](https://pypi.org/project/tyssue/) |

| Tests | Doc | CHAT |
| --- | --- | --- |
| [![Tests](https://github.com/vivarium-collective/vivatyssue/actions/workflows/tests.yml/badge.svg)](https://github.com/vivarium-collective/vivatyssue/actions/workflows/tests.yml) | [![Doc Status](https://readthedocs.org/projects/tyssue/badge/?version=latest)](http://tyssue.readthedocs.io/en/latest/) |  [![Join the chat at https://gitter.im/DamCB/tyssue](https://badges.gitter.im/DamCB/tyssue.svg)](https://gitter.im/DamCB/tyssue?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |


# tyssue is now published in the Journal of Open Source Software!

| ZENODO DOI | JOSS DOI |
| --- | --- |
| [![DOI](https://zenodo.org/badge/32533164.svg)](https://zenodo.org/badge/latestdoi/32533164) | [![DOI](https://joss.theoj.org/papers/10.21105/joss.02973/status.svg)](https://doi.org/10.21105/joss.02973) |

The `tyssue` library seeks to provide a unified interface to implement
biomechanical models of living tissues.
It's main focus is on **vertex based epithelium models**.

## Overview

### What kind of Models does it implement?

The first model implemented is the one described in
Monier et al.<sup id="a1">[1](#f1)</sup>. It is an example of a vertex model,
where the interactions are only evaluated on the apical surface sheet
of the epithelium. The second class of models is still at a
stage. They implement a description of the tissue's rheology, within a
dissipation function formalism.

![The two models considered](doc/illus/two_models.png)

### What kind of geometry does it model ?
`tyssue` allows to model different geometries in 2D and in 3D.  Presentation of the different geometries and how to create them are shown in this [notebook](https://github.com/DamCB/tyssue-demo/blob/master/01-Geometry.ipynb).
<p align="center">
 <img src="doc/illus/geometry.png" width="500">
</p>


### General Structure of the modeling API

#### Design principles

> [APIs not apps](https://opensource.com/education/15/9/apis-not-apps)

Each biological question, be it in morphogenesis or cancer studies is
unique, and requires tweeking of the models developed by the
physicists. Most of the modelling software follow an architecture
based on a core C++ engine with a combinaison of markup or scripting
capacities to run specific simulation.

In `tyssue`, we rather try to expose an API that simplifies the
building of tissue models and running simulations, while keeping the
possibilities as open as possible.

> Separate structure, geometry and models

We seek to have a design as modular as possible, to allow the same
epithelium mesh to be fed to different physical models.

> Accessible, easy to use data structures

The core of the tyssue library rests on two structures: a set of
`pandas DataFrame` holding the tissue geometry and associated data,
and nested dictionaries holding the model parameters, variables and
default values.

![Tyssue data structure](doc/illus/tyssue_data_management.png)

The API thus defines an `Epithelium` class. An instance of this class
is a container for the datasets and the specifications, and implements
methods to manipulate indexing of the dataframes to ease calculations.

The mesh structure is heavily inspired by
[CGAL Linear Cell Complexes](http://doc.cgal.org/latest/Linear_cell_complex/index.html),
most importantly, in the case of a 2D vertex sheet for example, each
junction edge between the cells is "splited" between two oriented **half
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

### Documentation

* The old documentation is still browsable online [here](http://tyssue.readthedocs.io/en/latest/)
* Introduction notebooks are available [here](https://github.com/DamCB/tyssue-demo).

### Quick view with Colab

To quickly try out and get an intution on how tyssue works, a jupyter notebook has been developed in [Google Colab](https://colab.research.google.com/github/vivarium-collective/vivatyssue/blob/main/notebooks/basic_usage_quick.ipynb).
Google Colab is an on-line jupyter notebook that requires no setup and is free to use.
To use it, simply sign in with your Google account and run all of the cells. 
Feel free to try different geometries, topologies, and mechanical properties that you can set within the Google Colab environment.

## Visualisation
Thanks to @kephale, there is a napari plugin to visualise tyssue simulation output.
You can find it [here](https://github.com/kephale/napari-tyssue).


### What's new in 1.0 ?
* Add new geometry : 2D lateral geometry
* Add mean calculation in `Epithelium`
* Add `sheet` conversion to `mesh` from `Meshio`
* Allow hole in 2D sheet
* Fix some visualisation

### Roadmap
You are welcome to participate in the development of `Tyssue`.
What is planned for the future of `Tyssue`?
* Solve collision in 2.5D & 3D
* Use ZARR instead of HDF5 as base file format
* Upgrade geometry creation
  * Compound geometry to form complex shape
  * Use biological image as blueprint
* Add rheology model

### Mailing list:

tyssue@framaliste.org - https://framalistes.org/sympa/info/tyssue

Subscribe ➙ https://framalistes.org/sympa/subscribe/tyssue
Unsubscribe ➙ https://framalistes.org/sympa/sigrequest/tyssue


### Authors

* Bertrand Caré - @bcare
* Cyprien Gay - @cypriengay
* Guillaume Gay - @glyg
* Hadrien Mary - @hadim
* François Molino
* Magali Suzanne
* Sophie Theis - @sophietheis
* Tasnif Rahman (maintainer) - @sniffo

## Dependencies

As all the dependencies are already completely supported in
python 3.x, **we won't be maintaining a python 2.x version**, because
it's time to move on...
<details>
  <summary>Show dependencies</summary>

### Core

- Python >= 3.10
- numpy
- scipy
- pandas
- matplotlib
- networkx
- meshio
- quantities
- scikit-image
- pytables
- ipywidgets

### Visualisation (`tyssue[viz]`)

- ipyvolume
- pythreejs
- vispy
- jupyter
- notebook

### Zarr (`tyssue[zarr]`)

- zarr
- xarray

### Tests

- pytest
- pytest-cov
- nbval
- nox

</details>


## Install

tyssue no longer depends on CGAL and is a pure python package, installed from PyPI:

```bash
python -m pip install --upgrade tyssue
```

### Optional extras

```bash
python -m pip install "tyssue[viz]"   # 3D and interactive visualisation
python -m pip install "tyssue[zarr]"  # zarr file support
```

The base install is enough to run simulations and to draw with matplotlib.
The `viz` extra is needed for the ipyvolume and vispy 3D views.

### Writing gifs

`create_gif` and `create_gif_3d` use [ImageMagick](https://imagemagick.org),
which is not a python package and has to be installed separately:

| Platform | Command |
| --- | --- |
| macOS | `brew install imagemagick` |
| Debian/Ubuntu | `sudo apt install imagemagick` |
| Windows | [installer](https://imagemagick.org/script/download.php#windows) |

### From source

See [INSTALL.md](INSTALL.md) for a step by step install.

## How to cite
If you find `tyssue` useful please cite the original publication and the
[upstream repository](https://github.com/DamCB/tyssue) using its DOI as follows:

> Theis, Suzanne, Gay, (2021). Tyssue: an epithelium simulation library. Journal of Open Source Software, 6(62), 2973 doi:[https://doi.org/10.21105/joss.02973](https://doi.org/10.21105/joss.02973)
>
> Zenodo doi:10.5281/zenodo.4817609

## Publications
If `tyssue` has made a substantial contribution to your work, please edit this file and open a PR.

Lou Y, Rupprecht JF, Theis S, Hiraiwa T, and Saunders TE. Curvature-induced cell rearrangements in biological tissues. Phys. Rev. Lett. 2023. doi: [10.1103/PhysRevLett.130.108401](https://doi.org/10.1103/PhysRevLett.130.108401)

Rahman T, Peters F and Wan LQ. Cell Jamming Regulates Epithelial Chiral Morphogenesis. J Biomech. 2023. doi: [10.1016/j.jbiomech.2023.111435](https://doi.org/10.1016/j.jbiomech.2023.111435)

Fiorentino J and Scialdone A. The role of cell geometry and cell-cell communication in gradient sensing. PLoS Comput Biol. 2022 doi: [10.1371/journal.pcbi.1009552](10.1371/journal.pcbi.1009552)
Related repository: [https://github.com/ScialdoneLab/2DLEGI](https://github.com/ScialdoneLab/2DLEGI)

Courcoubetis G, Xu C, Nuzhdin SV, Haas S. Avalanches during epithelial tissue growth; Uniform Growth and a drosophila eye disc model, PLoS Comput Biol 2022 doi: [10.1371/journal.pcbi.1009952](https://doi.org/10.1371/journal.pcbi.1009952)

Martin E, Theis S, Gay G, Monier B, Rouvière C, Suzanne M. Arp2/3-dependent mechanical control of morphogenetic robustness in an inherently challenging environment. Dev Cell. 2021 doi: [10.1016/j.devcel.2021.01.005](https://doi.org/10.1016/j.devcel.2021.01.005)
Related repository: [https://github.com/suzannelab/polarity](https://github.com/suzannelab/polarity)

Gracia M, Theis S, Proag A, Gay G, Benassayag C, Suzanne M. Mechanical impact of epithelial-mesenchymal transition on epithelial morphogenesis in Drosophila. Nat Commun. 2019 doi: [10.1038/s41467-019-10720-0](https://doi.org/10.1038/s41467-019-10720-0)
Related repository: [https://github.com/suzannelab/invagination](https://github.com/suzannelab/invagination)

Monier B, Gettings M, Gay G, Mangeat T, Schott S, Guarner A, Suzanne M. Apico-basal forces exerted by apoptotic cells drive epithelium folding. Nature. 2015 doi: [10.1038/nature14152](https://doi.org/10.1038/nature14152)
Related repository: [https://github.com/glyg/leg-joint](https://github.com/glyg/leg-joint)

## Licence

Since version 0.3, this project is distributed under the terms of the [General Public Licence](https://www.gnu.org/licenses/gpl.html).


Versions 2.4 and earlier were distributed under the [Mozilla Public Licence](https://www.mozilla.org/en-US/MPL/2.0/).

If GPL licencing is too restrictive for your intended usage, please contact the maintainer.

## Research notice
Please note that this repository is participating in a study into sustainability
 of open source projects. Data will be gathered about this repository for
 approximately the next 12 months, starting from June 2021.

Data collected will include number of contributors, number of PRs, time taken to
 close/merge these PRs, and issues closed.

For more information, please visit
[our informational page](https://sustainable-open-science-and-software.github.io/) or download our [participant information sheet](https://sustainable-open-science-and-software.github.io/assets/PIS_sustainable_software.pdf).

<hr/>

<b id="f1">1</b> : Monier B, Gettings M, Gay G, Mangeat T, Schott S, Guarner A, Suzanne M. Apico-basal forces exerted by apoptotic cells drive epithelium folding. Nature. 2015 doi: [10.1038/nature14152](https://doi.org/10.1038/nature14152) [↩](#a1)
