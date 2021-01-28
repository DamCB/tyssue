Since version 0.3, tyssue depends on CGAL for collision detection, and thus a c++ compiler toolchain. It is thus advised to use conda for a simple installation procedure.

## Installing tyssue with conda

If you have a conda environment ready:
```
conda install -c conda-forge tyssue
```

This will install tyssue and all its dependencies, with the pre-compiled binary parts.

## Install tyssue using pip

This install a cross-platform, pure python version of tyssue.
Some advanced features are not available, namely:

- Collision detection
- Periodic boundary sheet generation

```sh
python -m pip install --user --upgrade tyssue
```

## Installing from source

Those are the instructions to install the package from source on a
debian-like linux distribution. If you allready have a basic
scientific python stack, use it, don't install anaconda.

### Install a C++ compiler

With an Debian like system, this is achieved by:

```bash
sudo apt install build-essential cmake g++
```

### Download and install `tyssue` from source

If you want to do that, I assume you allready know how to manage
dependencies on your platform. The simplest way to manage dependencies is to use [`conda`](https://docs.conda.io/en/latest/miniconda.html) to manage the dependencies (you can use [`mamba`](https://github.com/mamba-org/mamba) as a faster alternative to conda).

Start by cloning tyssue recursively to also grab pybind11:

```bash
git clone --recursive https://github.com/damcb/tyssue.git
cd tyssue
```

Then create a virtual environement:

```bash
conda env create -f environment.yml
```

Then install python:
```
python setup.py install
```



If all went well, you have successfully installed tyssue.

### Install testing utilities

```sh
pip install pytest pytest-cov nbval
```

A `Makefile` provides some utility function. Try :

```sh
make tests  # Run tests with nose
make coverage  # Run tests with coverage
make flake8  # Check PEP8 on the code
make nbtest # Tests all  the demo notebooks - requires nbval
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
