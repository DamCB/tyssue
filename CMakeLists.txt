##############################################################################
# Use of cmake to install - good if you need to compile cython modules

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
# option(BUILD_SHARED_LIBS "Build shared library" ON)

## Build Option: when active headers will be installed in the CMAKE_INSTALL_PREFIX/include directory
# option(INSTALL_HEADERS "Installs the header files" ON)

## Build Option: build documentation
# option(BUILD_DOC "Build documentation." ON)
##############################################################################

# Find Python 3 (with numpy)
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibsNew 3 REQUIRED)
find_package(NumPy 3 REQUIRED)
find_package(SciPy REQUIRED)

# find_package(CGAL REQUIRED)
# message(STATUS "CGAL version: ${CGAL_VERSION}")

# if(NOT CGAL_FOUND OR ${CGAL_VERSION} VERSION_LESS 4.6)
#   message(FATAL_ERROR "Tyssue needs CGAL library >= 4.6.")
# endif()

##############################################################################

# Installation path

## By default installation prefix (CMAKE_INSTALL_PREFIX) will be set
## according to the path of the Python interpreter. This way it works out
## of the box with Anaconda installation.

get_filename_component(INSTALL_PREFIX "${PYTHON_LIBRARIES}" PATH)
get_filename_component(INSTALL_PREFIX "${INSTALL_PREFIX}" PATH)

if (NOT ${INSTALL_PREFIX} STREQUAL "")
  set(CMAKE_INSTALL_PREFIX "${INSTALL_PREFIX}")
endif()

message(STATUS "Install prefix: ${INSTALL_PREFIX}")

##############################################################################

# Add some flags to the compiler
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math -std=c++11")

##############################################################################

# add_subdirectory("${CMAKE_SOURCE_DIR}/src")

##############################################################################

if(BUILD_DOC)
    # add_subdirectory("${CMAKE_SOURCE_DIR}/doc")
endif(BUILD_DOC)

##############################################################################

message(STATUS "Tyssue will be installed to ${INSTALL_PREFIX}")


##############################################################################
# Formerly in src/CMakeLists.txt

# Install Python module
find_program(PYTHON "python" REQUIRED)

message(STATUS "Looking in  ${CMAKE_CURRENT_BINARY_DIR} for setup.py")


set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(DEPS     "${CMAKE_CURRENT_SOURCE_DIR}/tyssue/__init__.py")
set(OUTPUT   "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

# Run `python setup.py build`
add_custom_command(
  OUTPUT ${OUTPUT}
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} build
  COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
)

add_custom_target("Python tyssue module" ALL DEPENDS ${OUTPUT})

# Run `python setup.py install`
install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install
                              --prefix ${CMAKE_INSTALL_PREFIX}
                              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
