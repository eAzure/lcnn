aux_source_directory(../../src/tensor TENSOR_DIR)

add_executable(test_tensor_test test_tensor_test.cc ${TENSOR_DIR})
add_executable(test_tensor_construct test_tensor_construct.cc ${TENSOR_DIR})
add_executable(test_tensor test_tensor.cc ${TENSOR_DIR})

target_link_libraries(test_tensor_test ${GTEST_LIB} ${MATH_LIB})
target_link_libraries(test_tensor_construct ${GTEST_LIB} ${MATH_LIB})
target_link_libraries(test_tensor ${GTEST_LIB} ${MATH_LIB})

add_test(NAME test_tensor_test COMMAND test_tensor_test)
add_test(NAME test_tensor_construct COMMAND test_tensor_construct)
add_test(NAME test_tensor COMMAND test_tensor)