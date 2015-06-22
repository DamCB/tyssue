## Installing tyssue

Those are the instructions to install the package from source on a debian-like linux distribution.

### get Anaconda
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
conda create -n tyssue python=3.4 numpy scipy vispy matplotlib nose
## activate the new environment
source activate tyssue
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

That should work !
