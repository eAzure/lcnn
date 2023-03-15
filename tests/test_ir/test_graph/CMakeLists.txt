add_executable(test_graph test_graph.cc)

target_link_libraries(test_graph ${PROJECT_NAME} ${GTEST_LIB} ${MATH_LIB})

add_test(NAME test_graph COMMAND test_graph)