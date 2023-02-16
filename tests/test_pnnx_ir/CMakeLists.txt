aux_source_directory(../../src/pnnx_ir PNNX_IR_DIR)

add_executable(test_pnnx_ir test_pnnx_ir.cc ${PNNX_IR_DIR})

target_link_libraries(test_pnnx_ir ${GTEST_LIB})

add_test(NAME test_pnnx_ir COMMAND test_pnnx_ir)