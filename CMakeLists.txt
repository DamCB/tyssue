cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(tyssue)

##############################################################################

# Add custom CMake modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Some screen output
message(STATUS "OS detected: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CXX Compiler detected: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMake additional search path for libraries: ${CMAKE_LIBRARY_PATH}")


# CGAL and its components
find_package(CGAL QUIET COMPONENTS Core GMP MPFI)
message(STATUS "CGAL version: ${CGAL_VERSION}")

if (NOT CGAL_FOUND)
  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()
endif()


# include helper file
include( ${CGAL_USE_FILE})


set (SOURCE_DIR "tyssue/collisions/cpp")
include_directories (${SOURCE_DIR})

# Assumes pybind11 is a subdirectory at the project root
# Make sure you git cloned tyssue recursively
add_subdirectory(pybind11)
pybind11_add_module(c_collisions ${SOURCES} "${SOURCE_DIR}/c_collisions.cpp" )
