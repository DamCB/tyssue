# Install Python module
find_program(PYTHON "python" REQUIRED)

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
