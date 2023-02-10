cmake_minimum_required(VERSION 3.16)
project(lcnn)
set(CMAKE_CXX_STANDARD 11)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR}/include)

enable_testing()

add_subdirectory(3rdparty/googletest)
add_subdirectory(3rdparty/spdlog)

add_subdirectory(tests)