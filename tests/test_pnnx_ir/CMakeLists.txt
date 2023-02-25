add_executable(test_pnnx_ir test_pnnx_ir.cc)

target_link_libraries(test_pnnx_ir ${PROJECT_NAME} ${GTEST_LIB} ${MATH_LIB})

add_test(NAME test_pnnx_ir COMMAND test_pnnx_ir)