add_executable(test_operator_attribute test_operator_attribute.cc)
add_executable(test_operator_parameter test_operator_parameter.cc) 

target_link_libraries(test_operator_attribute ${PROJECT_NAME} ${GTEST_LIB} ${MATH_LIB})
target_link_libraries(test_operator_parameter ${PROJECT_NAME} ${GTEST_LIB} ${MATH_LIB})

add_test(NAME test_operator_attribute COMMAND test_op_relu)
add_test(NAME test_operator_parameter COMMAND test_op_register)