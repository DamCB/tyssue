## Installing tyssue

Those are the instructions to install the package from source on a
debian-like linux distribution. If you allready have a basic
scientific python stack, use it, don't install anaconda.


### Get Anaconda
Go to http://continuum.io/downloads and grab anaconda for your architecture.

### Install Anaconda

```bash
bash Anaconda3-2.2.0-Linux-x86_64.sh
source .bashrc # update your PATH
```

### Install package dependencies

```bash
sudo apt-get install git
```

### Create a virtual environment with `conda`

```bash
conda create -n tyssue python=3.4 numpy scipy vispy matplotlib nose coverage
## activate the new environment
source activate tyssue
## install some friends
conda install -c https://conda.anaconda.org/osgeo cgal
```

### Download and complie `tyssue`

```bash
git clone https://github.com/CellModels/tyssue.git
cd tyssue/
mkdir build/ && cd build/
cmake ..
make && make install
```

If all went well, you have successfully installed tyssue.

A `Makefile` provides some utility function. Try :

```sh
make flake8  # Check PEP8 on the code
make tests  # Run tests with nose
make coverage  # Run tests with coverage
```

### Conda build

You can build a conda binary :

```sh
conda build conda-recipe/
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
