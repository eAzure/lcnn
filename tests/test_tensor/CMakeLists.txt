aux_source_directory(../../src/tensor TENSOR_DIR)

add_executable(test_tensor_test test_tensor_test.cc ${TENSOR_DIR})

target_link_libraries(test_tensor_test ${GTEST_LIB})

add_test(NAME test_tensor_test COMMAND test_tensor_test)
