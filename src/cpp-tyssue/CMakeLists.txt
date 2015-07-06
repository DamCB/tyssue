# Global settings for include paths
include_directories(
  ${PROJECT_SOURCE_DIR}/src/cpp-tyssue/include
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIRS}
  ${SCIPY_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CGAL_INCLUDE_DIRS}
)

# Retrieve source code and headers
file(GLOB_RECURSE header_files ${PROJECT_SOURCE_DIR}/src/cpp-tyssue/include/ *.hh)
file(GLOB_RECURSE source_files ${PROJECT_SOURCE_DIR}/src/cpp-tyssue/ *.cc)

add_library(tyssue_core ${LIBRARY_TYPE}
            ${header_files}
            ${source_files}
            )

target_link_libraries(tyssue_core
                      ${Boost_LIBRARIES}
                      ${PYTHON_LIBRARIES}
                      ${NUMPY_LIBRARIES}
                      ${CGAL_LIBRARY}
                      )

install(TARGETS tyssue_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Install header files
if(INSTALL_HEADERS)
    INSTALL(FILES ${header_files} DESTINATION "include/tyssue")
endif(INSTALL_HEADERS)
