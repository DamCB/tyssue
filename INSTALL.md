## Installing tyssue

Those are the instructions to install the package from source on a debian-like linux distribution


### get Anaconda
Go to http://continuum.io/downloads and grab anaconda for your architecture.

### Install Anaconda

```bash
  bash Anaconda3-2.2.0-Linux-x86_64.sh
  source .bashrc ## updates your PATH
```

### Install package dependencies

```bash
  sudo apt-get install build-essential
  sudo apt-get install libexpat1-dev
  sudo apt-get install libboost1.54-all-dev
  sudo apt-get install libcgal-dev
  sudo apt-get install git
  sudo apt-get install automake autoconf
  sudo apt-get install libtool
```

### Install spareshash

```bash
  wget https://sparsehash.googlecode.com/files/sparsehash_2.0.2-1_amd64.deb
  sudo dpkg -i sparsehash_2.0.2-1_amd64.deb
```

### Create a virtual environment with `conda`

```bash
  conda create -n tyssue_env python=3.4 anaconda
  ## activate the new environment
  source activate tyssue_env
```

### Download and complie `tyssue`

```bash
  git clone https://github.com/CellModels/tyssue.git
  cd tyssue/
  which python ## outputs the current python binary
  export PYTHON={replace by what was output by which}
  export PYTHON_VERSION=3.4
  ./autogen.sh
  ./configure --with-boost-python=py34
  make
  sudo make install ## sudo should not be needed in the future
```

If all went well, you have successfully install tyssue

You can now start a python interpreter (by typing `python` in the command line) and run:

```python
  >>> import tyssue
  >>> tyssue.core.test_import #should print 'howdy'
```

That should work!
