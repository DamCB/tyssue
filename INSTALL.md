## Installing tyssue

Those are the instructions to install the package from source on a debian-like linux distribution.

### Get Anaconda
Go to http://continuum.io/downloads and grab anaconda for your architecture.

### Install Anaconda

```bash
bash Anaconda3-2.2.0-Linux-x86_64.sh
source .bashrc # update your PATH
```

### Install package dependencies

```bash
sudo apt-get install build-essential
sudo apt-get install libexpat1-dev
sudo apt-get install git
```

### Install spareshash

```bash
wget https://sparsehash.googlecode.com/files/sparsehash_2.0.2-1_amd64.deb
sudo dpkg -i sparsehash_2.0.2-1_amd64.deb
```

### Create a virtual environment with `conda`

```bash
conda create -n tyssue python=3.4 numpy scipy vispy matplotlib nose coverage
## activate the new environment
source activate tyssue
## install some friends
conda install -c https://conda.binstar.org/osgeo boost cgal
pip install vispy
```

### Download and complie `tyssue`

```bash
git clone https://github.com/CellModels/tyssue.git
cd tyssue/
mkdir build/ && cd build/
cmake ..
make && make install
```

If all went well, you have successfully installed tyssue. To test it, you can run `python -c "import tyssue; tyssue.core.test_import()"`. It should print `howdy`.

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
