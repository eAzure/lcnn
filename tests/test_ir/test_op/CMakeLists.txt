aux_source_directory(../../../src/tensor TENSOR_DIR)
aux_source_directory(../../../src/pnnx_ir PNNX_IR_DIR)
# aux_source_directory(../../../src/ir/op OP_DIR)
# aux_source_directory(../../../src/ir/op/ops OPS_DIR)

file(GLOB_RECURSE IR_SOURCES "../../../src/ir/*.cc" )

# add_executable(test_op_relu test_op_relu.cc ${TENSOR_DIR} ${OP_DIR} ${OPS_DIR})
add_executable(test_op_relu test_op_relu.cc ${TENSOR_DIR} ${PNNX_IR_DIR} ${IR_SOURCES})
add_executable(test_op_register test_op_register.cc ${TENSOR_DIR} ${PNNX_IR_DIR} ${IR_SOURCES})

# [TODO] 搞清除为什么添加这个就不会报"undefined reference to ..."
link_directories(${TENSOR_DIR} ${PNNX_IR_DIR} ${IR_SOURCES})

target_link_libraries(test_op_relu ${GTEST_LIB} ${MATH_LIB})
target_link_libraries(test_op_register ${GTEST_LIB} ${MATH_LIB})

add_test(NAME test_op_relu COMMAND test_op_relu)
add_test(NAME test_op_register COMMAND test_op_register)