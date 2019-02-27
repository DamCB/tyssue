Since version 0.3, tyssue depends on CGAL for collision detection, and thus a c++ compiler toolchain. It is thus advised to use conda for a simple installation procedure.

## Installing tyssue with conda

If you already have a conda environment ready:
```
conda install -c conda-forge tyssue
```

This will install tyssue and all its dependencies, with the pre-compiled binary parts.


## Installing from source

Those are the instructions to install the package from source on a
debian-like linux distribution. If you allready have a basic
scientific python stack, use it, don't install anaconda.

### Install a C++ compiler

With an Debian like system, this is achieved by:

```bash
sudo apt install build-essential cmake g++
```


### Install CGAL

You can use a pre-packaged version for your OS. For Debian:
```bash
sudo apt install libcgal-dev
```
Note that you need version 4.7 or higher.

If you need to install CGAL from source, here is a `cmake` command sufficient to later compile tyssue:

```bash
cd CGAL # This is the directory where the CGAL archive was uncompressed
mkdir build && cd build

# needs qt5 for imageio
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=${PREFIX} \
  -DCMAKE_INSTALL_PREFIX=${PREFIX} \
  -DCGAL_INSTALL_LIB_DIR=lib \
  -DWITH_CGAL_ImageIO=OFF -DWITH_CGAL_Qt5=OFF \
  ..
make install -j${CPU_COUNT}
```


### Download and install `tyssue` from source

If you want to do that, I assume you allready know how to manage
dependencies on your platform.

```bash
git clone --recursive https://github.com/damcb/tyssue.git
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
