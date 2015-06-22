find_program(PYTHON "python" REQUIRED)

set(SETUP_PY "${PROJECT_SOURCE_DIR}/setup.py")

install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})")
