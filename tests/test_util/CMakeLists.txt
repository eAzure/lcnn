add_executable(test_gtest test_gtest.cc)
target_link_libraries(test_gtest ${GTEST_LIB})

add_test(NAME test_gtest COMMAND test_gtest)