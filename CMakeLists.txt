cmake_minimum_required(VERSION 2.8)
project(Tyssue)

##############################################################################

# Add custom CMake modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# The version number
# TODO: set it from the source code
set(Tyssue_VERSION_MAJOR 0)
set(Tyssue_VERSION_MINOR 1)
set(Tyssue_VERSION ${Tyssue_VERSION_MAJOR}.${Tyssue_VERSION_MINOR})

# Some screen output
message(STATUS "OS detected: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CXX Compiler detected: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMake additional search path for libraries: ${CMAKE_LIBRARY_PATH}")

##############################################################################

# Provides build options to CMake

## Build option: build shared library (should be automatically put to static for win32)
option(BUILD_SHARED_LIBS "Build shared library" ON)

## Build Option: when active headers will be installed in the CMAKE_INSTALL_PREFIX/include directory
option(INSTALL_HEADERS "Installs the header files" ON)

## Build Option: build documentation
option(BUILD_DOC "Build documentation." ON)

##############################################################################

# Find boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost 1.54 COMPONENTS python3 QUIET)
if(NOT Boost_FOUND)
  find_package(Boost 1.54 COMPONENTS python-py34 REQUIRED)
endif()
message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")

# Find Python 3 (with numpy)
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibsNew 3 REQUIRED)
find_package(NumPy 3 REQUIRED)
find_package(SciPy REQUIRED)

# Find CGAL
find_package(CGAL REQUIRED)

##############################################################################

# Add some flags to the compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math")

##############################################################################

add_subdirectory("${CMAKE_SOURCE_DIR}/src")

##############################################################################

if(BUILD_DOC)
    # add_subdirectory("${CMAKE_SOURCE_DIR}/doc")
endif(BUILD_DOC)

##############################################################################

# Installation path

## By default installation prefix (CMAKE_INSTALL_PREFIX) will be set
## according to the path of the Python interpreter. This way it works out
## of the box with Anaconda installation.

get_filename_component(INSTALL_PREFIX "${PYTHON_LIBRARIES}" PATH)
get_filename_component(INSTALL_PREFIX "${INSTALL_PREFIX}" PATH)

set(CMAKE_INSTALL_PREFIX "${INSTALL_PREFIX}")
message(STATUS "Tyssue will be installed to ${INSTALL_PREFIX}")
