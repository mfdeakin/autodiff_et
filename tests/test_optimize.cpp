
#include <cmath>

#include "autodiff.hpp"
#include "autodiff_optimize.hpp"
#include "autodiff_transcendental.hpp"

#include <random>

#include <gtest/gtest.h>

using namespace auto_diff;

using rngAlg = std::mt19937_64;

TEST(random_test, zoom_ls) {
  constexpr variable<double> x_1(0);
  constexpr variable<double> x_2(1);
  constexpr double x_min = -9.0;
  constexpr double y_min = 42.0;
  constexpr auto e =
      x_1 * x_1 + x_1 * x_2 + x_2 * x_2 - 24.0 * x_1 - 75.0 * x_2 - 432.0;
  // c1 and c2 must be between 0 and 1, in increasing order
  constexpr double c1 = 0.25, c2 = 0.75;

  CNLMin optimizer(e);
  std::vector<double> new_pt(2);
  std::vector<double> mini_dir(2);

  rngAlg engine((std::random_device())());
  const double pdf_bound = 128.0;
  std::uniform_real_distribution<double> pdf(-pdf_bound, pdf_bound);
  for (int i = 0; i < 10; ++i) {
    // Ensure the line search satisfies the strong Wolfe conditions
    // That is,
    // 1) f(x_k + alpha p_k) <= f(x_k) + c_1 \alpha grad(f)(x_k)^T p_k
    // 2) grad(f)(x_k + alpha p_k)^T p_k >= c_2 grad(f)(x_k)^T p_k
    // where x_k is the initial point, p_k is the search direction,
    // \alpha is the step length, c_1 and c_2 are increasing constants in (0, 1)
    const std::vector<double> x0{pdf(engine), pdf(engine)};
    // Choose a search direction in a direction close to the gradient direction
    optimizer.grad_search_dir(x0, mini_dir);
    const double theta = pdf(engine) * M_PI / (2.0 * pdf_bound) * 2.0 / 5.0;
    const std::vector<double> search_dir{
        (mini_dir[0] * std::cos(theta) + mini_dir[1] * std::sin(theta)),
        (-mini_dir[0] * std::sin(theta) + mini_dir[1] * std::cos(theta))};

    const double dx = x_min - x0[0], dy = y_min - x0[1],
                 dmag = std::sqrt(dx * dx + dy * dy);
    const double deriv_val = optimizer.dir_deriv(x0, search_dir);
    const double min_step = 0.0009765625 * dmag, max_step = 2.0 * dmag;
    const double step_size =
        optimizer.zoom(x0, search_dir, min_step, max_step, c1, c2, new_pt);
    ASSERT_FALSE(std::isnan(step_size));

    optimizer.grad_search_dir(new_pt, mini_dir);
    const double new_deriv =
        -(search_dir[0] * mini_dir[0] + search_dir[1] * mini_dir[1]);
    // Verify the point is within the required step range
    ASSERT_LE(step_size, max_step);
    ASSERT_GE(step_size, min_step);
    // Check the Wolfe conditions
    ASSERT_LE(e.eval(new_pt), e.eval(x0) + c1 * step_size * deriv_val);
    ASSERT_GE(new_deriv, c2 * deriv_val);
  }
}

TEST(fixed_test, wolfe_ls) {
  constexpr variable<double> x_1(0);
  constexpr variable<double> x_2(1);
  constexpr auto e =
      x_1 * x_1 + x_1 * x_2 + x_2 * x_2 - 24.0 * x_1 - 75.0 * x_2 - 432.0;
  // e has a minimum at (-9, 42)
  // constexpr double x_min = -9.0;
  // constexpr double y_min = 42.0;

  // Ensure the search direction is just the negative gradient
  CNLMin optimizer(e);
  std::vector<double> mini_dir(2);
  optimizer.grad_search_dir({0.0, 0.0}, mini_dir);

  // Ensure the line search satisfies the strong Wolfe conditions
  // That is,
  // 1) f(x_k + alpha p_k) <= f(x_k) + c_1 \alpha grad(f)(x_k)^T p_k
  // 2) grad(f)(x_k + alpha p_k)^T p_k >= c_2 grad(f)(x_k)^T p_k
  const std::vector<double> x0{0.0, 0.0};
  const std::vector<double> search_dir{1.0, 4.0};
  std::vector<double> new_pt(2);
  optimizer.grad_search_dir(x0, mini_dir);
  const double deriv_val =
      -(search_dir[0] * mini_dir[0] + search_dir[1] * mini_dir[1]);
  // c1 and c2 must be between 0 and 1, in increasing order
  const double c1 = 0.25, c2 = 0.75;
  const double max_step = 4.0;
  const double step_size =
      optimizer.strong_wolfe_ls(x0, search_dir, new_pt, max_step);
  ASSERT_FALSE(std::isnan(step_size));

  // Verify the point is within the required step range
  ASSERT_GE(step_size, 0.0);
  ASSERT_LE(step_size, max_step);

  optimizer.grad_search_dir(new_pt, mini_dir);
  double new_deriv =
      -(search_dir[0] * mini_dir[0] + search_dir[1] * mini_dir[1]);
  // Check the Wolfe conditions
  ASSERT_LE(e.eval(new_pt), c1 * step_size * deriv_val);
  ASSERT_GE(new_deriv, c2 * deriv_val);
}

TEST(random_test, strong_wolfe_ls) {
  constexpr variable<double> x_1(0);
  constexpr variable<double> x_2(1);
  constexpr double x_min = -9.0;
  constexpr double y_min = 42.0;
  constexpr auto e =
      x_1 * x_1 + x_1 * x_2 + x_2 * x_2 - 24.0 * x_1 - 75.0 * x_2 - 432.0;
  // c1 and c2 must be between 0 and 1, in increasing order
  constexpr double c1 = 0.25, c2 = 0.75;

  CNLMin optimizer(e);
  std::vector<double> new_pt(2);
  std::vector<double> mini_dir(2);

  rngAlg engine((std::random_device())());
  const double pdf_bound = 128.0;
  std::uniform_real_distribution<double> pdf(-pdf_bound, pdf_bound);
  for (int i = 0; i < 10; ++i) {
    // Ensure the line search satisfies the strong Wolfe conditions
    // That is,
    // 1) f(x_k + alpha p_k) <= f(x_k) + c_1 \alpha grad(f)(x_k)^T p_k
    // 2) grad(f)(x_k + alpha p_k)^T p_k >= c_2 grad(f)(x_k)^T p_k
    // where x_k is the initial point, p_k is the search direction,
    // \alpha is the step length, c_1 and c_2 are increasing constants in (0, 1)
    const std::vector<double> x0{pdf(engine), pdf(engine)};
    // Choose a search direction in a direction close to the gradient direction
    optimizer.grad_search_dir(x0, mini_dir);
    const double dx = x_min - x0[0], dy = y_min - x0[1],
                 dmag = std::sqrt(dx * dx + dy * dy), max_step = 2.0 * dmag;
    const double theta = pdf(engine) * M_PI / (2.0 * pdf_bound) * 2.0 / 5.0;
    const std::vector<double> search_dir{
        (mini_dir[0] * std::cos(theta) + mini_dir[1] * std::sin(theta)),
        (-mini_dir[0] * std::sin(theta) + mini_dir[1] * std::cos(theta))};

    const double deriv_val = optimizer.dir_deriv(x0, search_dir);
    const double step_size =
        optimizer.strong_wolfe_ls(x0, mini_dir, new_pt, max_step);
    ASSERT_FALSE(std::isnan(step_size));

    optimizer.grad_search_dir(new_pt, mini_dir);
    const double new_deriv =
        -(search_dir[0] * mini_dir[0] + search_dir[1] * mini_dir[1]);
    // Verify the point is within the required step range
    ASSERT_LE(step_size, max_step);
    // Check the Wolfe conditions
    ASSERT_LE(e.eval(new_pt), e.eval(x0) + c1 * step_size * deriv_val);
    ASSERT_GE(new_deriv, c2 * deriv_val);
  }
}
