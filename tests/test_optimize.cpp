
#include <cmath>

#include "autodiff.hpp"
#include "autodiff_optimize.hpp"
#include "autodiff_transcendental.hpp"

#include <random>

#include <gtest/gtest.h>

using namespace auto_diff;

using rngAlg = std::mt19937_64;

TEST(cg_test, autodiff_optimize) {
  constexpr double eps = std::numeric_limits<double>::epsilon() * 128.0;
  constexpr variable<double> x_1(0);
  constexpr variable<double> x_2(1);
  constexpr double x_min = -9.0;
  constexpr double y_min = 42.0;
  constexpr auto e =
      x_1 * x_1 + x_1 * x_2 + x_2 * x_2 - 24.0 * x_1 - 75.0 * x_2 - 432.0;

  CNLMin optimizer(e);
  std::vector<double> search_dir(2);
  optimizer.grad_search_dir({0.0, 0.0}, search_dir);
  EXPECT_EQ(search_dir[0], 24.0);
  EXPECT_EQ(search_dir[1], 75.0);
  optimizer.grad_search_dir({1.0, 0.0}, search_dir);
  EXPECT_EQ(search_dir[1], 74.0);
  EXPECT_EQ(e.deriv(x_1.id()).eval({1.0}), -22.0);
  EXPECT_EQ(e.deriv(x_2.id()).eval({1.0}), -74.0);

  optimizer.grad_search_dir({1.0, 0.0}, search_dir);
  EXPECT_EQ(search_dir[1], 74.0);
  EXPECT_EQ(e.deriv(x_1.id()).eval({2.0}), -20.0);
  EXPECT_EQ(e.deriv(x_2.id()).eval({2.0}), -73.0);

  std::random_device rd;
  rngAlg engine(rd());
  std::uniform_real_distribution<double> pdf(-1024.0, 1024.0);
  for (int i = 0; i < 100; ++i) {
    const double x0_1 = pdf(engine);
    const double x0_2 = pdf(engine);
    optimizer.grad_search_dir({x0_1, x0_2}, search_dir);
    EXPECT_EQ(search_dir[0],
              -e.deriv(x_1.id()).eval(x_1.id(), x0_1, x_2.id(), x0_2));
    EXPECT_EQ(search_dir[1],
              -e.deriv(x_2.id()).eval(x_1.id(), x0_1, x_2.id(), x0_2));
  }

  const std::vector<double> &mini = optimizer.local_minimum();
  ASSERT_EQ(mini.size(), 2);
  EXPECT_NEAR(mini.at(0), x_min, eps);
  EXPECT_NEAR(mini.at(1), y_min, eps);
}
