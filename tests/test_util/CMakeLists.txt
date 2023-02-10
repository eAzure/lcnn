set(GTEST_LIB gtest gtest_main)

add_executable(test_gtest test_gtest.cc)
target_link_libraries(test_gtest ${GTEST_LIB})

add_executable(test_spdlog test_spdlog.cc)
target_link_libraries(test_spdlog ${GTEST_LIB} spdlog::spdlog)

add_test(NAME test_gtest COMMAND test_gtest)