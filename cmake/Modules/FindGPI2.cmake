# Use this to find the GPI library and headers.
#
# This module searches for the GPI and the GPI daemon libraries.
# The first directory it checks is /prog/GPI/, then all the standard locations.
#
# It sets the following variables:
#
#  GPI_FOUND - set to TRUE if GPI has been found successfully
#  GPI_INCLUDE_DIR - the include directory for GPI. Use this with include_directories()
#  GPI_GPI_LIBRARIES - contains the libraries you need to link against when using GPI.
#  GPI_GPIDAEMON_LIBRARIES - contains the libraries you need to link against when using the GPIDaemon library.
#
# Example usage:
# find_package(GPI REQUIRED)
#
# include_directories(${GPI_INCLUDE_DIR})
# add_executable(foo main.cpp)
# target_link_libraries(foo ${GPI_GPI_LIBRARIES} )

string(REPLACE ":" ";" SEARCH_DIRS "$ENV{LD_LIBRARY_PATH}")


set (SEARCH_PREFIX "/p/hpc/GPI" "${SEARCH_DIRS}")
message(STATUS "SEARCHING FOR GPI2:${SEARCH_PREFIX}")

find_path(GPI2_INCLUDE_DIR NAMES GASPI.h PATHS ${SEARCH_PREFIX} PATH_SUFFIXES include)

find_library(GPI2_GPI_LIBRARY GPI2 HINTS ${SEARCH_PREFIX} PATH_SUFFIXES lib64 )

find_library(GPI2_IBVERBS_LIBRARY ibverbs HINTS ${SEARCH_PREFIX} PATH_SUFFIXES lib64 )

#message(STATUS "A:${GPI2_GPI_LIBRARY} B:${GPI2_IBVERBS_LIBRARY} C:${GPI2_INCLUDE_DIR}")

set(GPI2_GPI_LIBRARIES ${GPI2_GPI_LIBRARY} ${GPI2_IBVERBS_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GPI2  DEFAULT_MSG  GPI2_GPI_LIBRARY GPI2_INCLUDE_DIR )

