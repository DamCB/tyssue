#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Make build dir
mkdir -p build
cd build/
rm -fr *

# CMake
cmake ..

make
make install

cd ../
make test
