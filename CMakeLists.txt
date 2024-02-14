cmake_minimum_required(VERSION 3.15...3.26)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
message(STATUS '${SKBUILD_PROJECT_NAME}')
set(CMAKE_VERBOSE_MAKEFILE ON)


set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
set(Python_LIBRARY "${PYTHON_LIBRARY}")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

find_package(Python COMPONENTS Interpreter Development.Embed REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)


# Some screen output
message(STATUS "OS detected: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CXX Compiler detected: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMake additional search path for libraries: ${CMAKE_LIBRARY_PATH}")

# CGAL and its components
find_package(CGAL)
message(STATUS "CGAL version: ${CGAL_VERSION}")

if (NOT CGAL_FOUND)
  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()
endif()

# include helper file
include( ${CGAL_USE_FILE})
message(STATUS "CGAL_USE_FILE: ${CGAL_USE_FILE}")

set (CPP_SRC "src/tyssue_cpp/")


pybind11_add_module(_mesh_generation MODULE "${CPP_SRC}/mesh_generation.cpp" )
install(TARGETS _mesh_generation DESTINATION ${SKBUILD_PROJECT_NAME})

pybind11_add_module(_collisions SHARED "${CPP_SRC}/c_collisions.cpp" )
install(TARGETS _collisions DESTINATION ${SKBUILD_PROJECT_NAME})
