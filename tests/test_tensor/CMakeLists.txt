add_executable(test_tensor_test test_tensor_test.cc)
add_executable(test_tensor_construct test_tensor_construct.cc)
add_executable(test_tensor test_tensor.cc)

target_link_libraries(test_tensor_test ${PROJECT_NAME} ${GTEST_LIB} ${MATH_LIB})
target_link_libraries(test_tensor_construct ${PROJECT_NAME} ${GTEST_LIB} ${MATH_LIB})
target_link_libraries(test_tensor ${PROJECT_NAME} ${GTEST_LIB} ${MATH_LIB})

add_test(NAME test_tensor_test COMMAND test_tensor_test)
add_test(NAME test_tensor_construct COMMAND test_tensor_construct)
add_test(NAME test_tensor COMMAND test_tensor)