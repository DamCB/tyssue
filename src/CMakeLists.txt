# Compile and install C++ lib
add_subdirectory("cpp-tyssue")

# Install Python module
find_program(PYTHON "python" REQUIRED)

set(SETUP_PY_IN        "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY           "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
# set(DEPS               "${CMAKE_CURRENT_SOURCE_DIR}/tyssue/__init__.py")
# set(OUTPUT             "${CMAKE_CURRENT_BINARY_DIR}/build")

# Generate setup.py
configure_file(${SETUP_PY_IN} ${SETUP_PY})

# # Run `python setup.py -q build`
# add_custom_command(
#   OUTPUT ${OUTPUT}/timestamp
#   COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY}
#   ARGS  build
#   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
#   DEPENDS ${DEPS} ${SETUP_PY_IN}
# )
# add_custom_target("Tyssue Python module" ALL DEPENDS ${OUTPUT}/timestamp)

# Run `python setup.py install`
install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install)")
