#include "gtest/gtest.h"
#include "spdlog/spdlog.h"

namespace {

TEST(UtilsTest, test_spdlog) {
    spdlog::info("Welcome to spdlog!");
}

} // namespace