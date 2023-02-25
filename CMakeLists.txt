cmake_minimum_required(VERSION 3.16)
project(lcnn)
set(CMAKE_CXX_STANDARD 11)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(MATH_LIB armadillo blas lapack)

enable_testing()

add_subdirectory(3rdparty/googletest)

file(GLOB_RECURSE LCNN_SOURCES "src/*.cc")
add_library(${PROJECT_NAME} SHARED ${LCNN_SOURCES})

add_subdirectory(tests)