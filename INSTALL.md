## Installing tyssue with conda

If you already have a conda environment ready:
```
conda install -c conda-forge tyssue
```

## From the beginnig

Those are the instructions to install the package from source on a
debian-like linux distribution. If you allready have a basic
scientific python stack, use it, don't install anaconda.


### Get and install Anaconda

Go to http://continuum.io/downloads and grab anaconda for your
architecture, and follow up to date instructions there.x



If you don't want the whole distrib, you can alternatively download
[miniconda](http://conda.pydata.org/miniconda.html)

### Create a virtual environment with `conda`

```bash
conda env create -f environment.yml # installs the dependencies
## activate the new environment
source activate tyssue
```


### Download and install `tyssue` from source

If you want to do that, I assume you allready know how to manage
dependencies on your platform.

```bash
git clone https://github.com/CellModels/tyssue.git
cd tyssue/
python setup.py install
```

If all went well, you have successfully installed tyssue.

A `Makefile` provides some utility function. Try :

```sh
make tests  # Run tests with nose
make coverage  # Run tests with coverage
make flake8  # Check PEP8 on the code
```

### Building the documentation

The documentation uses
[nbsphinx](http://nbsphinx.readthedocs.io/en/0.2.9/index.html) to
convert the jupyter notebooks in doc/notebooks to html with sphinx.


```sh
pip install sphinx nbsphinx sphinx-autobuild
cd tyssue/doc
make html
```
