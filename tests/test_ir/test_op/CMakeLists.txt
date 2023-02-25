add_executable(test_op_relu test_op_relu.cc)
add_executable(test_op_register test_op_register.cc) 

# 库依赖顺序 前面的库依赖后面的库
target_link_libraries(test_op_relu ${PROJECT_NAME} ${GTEST_LIB} ${MATH_LIB})
target_link_libraries(test_op_register ${PROJECT_NAME} ${GTEST_LIB} ${MATH_LIB})

add_test(NAME test_op_relu COMMAND test_op_relu)
add_test(NAME test_op_register COMMAND test_op_register)