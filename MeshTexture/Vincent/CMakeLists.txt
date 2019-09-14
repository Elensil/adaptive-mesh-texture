cmake_minimum_required(VERSION 2.8)

#-- Set the output path that will store the exexutables and binaries
SET( LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin )
SET( EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

#-- Add the folder containing the additional cmake tools.
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ")


#-- add subdirectories
add_subdirectory(Lib)
add_subdirectory(Core)

