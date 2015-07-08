ENV_NAME="$(date +"%y-%m-%d")-tyssue-test"
conda create --yes -n $ENV_NAME python=3

source activate $ENV_NAME

conda install --yes numpy scipy vispy matplotlib nose coverage vispy pip
conda install --yes -c https://conda.binstar.org/osgeo boost cgal

mkdir -p build/
cd build/
rm -fr *
cmake ..
make install

cd ../
make coverage

source deactivate
conda env remove -n $ENV_NAME --yes
