
#include <cmath>

#include "autodiff.hpp"
#include "autodiff_optimize.hpp"
#include "autodiff_tao.hpp"
#include "autodiff_transcendental.hpp"

#include "petsc_helpers.hpp"

#include <random>

#include <gtest/gtest.h>

#include "SyPDSolver.hpp"

using namespace auto_diff;

using rngAlg = std::mt19937_64;

TEST(polynomial_optimize, autodiff) {
  constexpr variable<double> x(0), y(1);
  std::random_device rd;
  rngAlg engine(rd());
  std::uniform_real_distribution<double> pdf(-1024.0, 1024.0);
  for (int tests = 0; tests < 100; ++tests) {
    const double cxx = abs(pdf(engine)) + 1e-4;
    const double cyy = abs(pdf(engine)) + 1e-4;
    // Ensure we always have a global minimum, by ensuring that
    // cxx > 0.0, and 4 * cxx * cyy - cxy ** 2 > 0
    // Let |cxy| < 2 * min(cxx, cyy)
    // Then cxy ** 2 < 4 * min(cxx, cyy) ** 2 < 4 * cxx * cyy
    const double cxy = abs(pdf(engine)) / 1024.0 * 2.0 * std::min(cxx, cyy);

    const double cx = pdf(engine);
    const double cy = pdf(engine);

    const double c = pdf(engine);

    double x_val = 0.0;
    double y_val = 0.0;

    const auto eqn =
        cxx * x * x + cxy * x * y + cyy * y * y + cx * x + cy * y + c;
    const auto soln_expr = gradient(eqn);
    SyPDSolver solver(soln_expr.size());
    std::vector<double> soln(soln_expr.size());
    const auto system = hessian(eqn);

    double dist = std::numeric_limits<double>::infinity();
    while (dist > 1e-10) {
      for (auto_diff::id_t i = 0; i < soln.size(); ++i) {
        for (auto_diff::id_t j = 0; j < soln.size(); ++j) {
          const double v = system.at(std::pair{i, j});
          solver(i, j, v);
        }
        soln[i] = soln_expr.at(i).eval(x.id(), x_val, y.id(), y_val);
      }
      std::vector<double> delta = solver.solve(soln);
      x_val -= delta[0];
      y_val -= delta[1];

      dist = delta[0] * delta[0] + delta[1] * delta[1];
    }
    for (auto_diff::id_t i = 0; i < soln.size(); ++i) {
      EXPECT_NEAR(soln_expr.at(i).eval(x.id(), x_val, y.id(), y_val), 0.0,
                  1e-10);
    }
  }
}

constexpr auto rosenbrock_function_term(const double &alpha,
                                        const auto_diff::variable<double> &x0,
                                        const auto_diff::variable<double> &x1) {
  return alpha * (x1 - x0 * x0) * (x1 - x0 * x0) + (1.0 - x0) * (1.0 - x0);
}

template <size_t dim, typename expr_t>
constexpr auto rosenbrock_function_helper(const expr_t &e, const double &alpha,
                                          const auto_diff::id_t var_id) {
  const auto_diff::variable<double> x0(var_id), x1(var_id + 1);
  const auto new_e = e + rosenbrock_function_term(alpha, x0, x1);
  // More inefficient than it needs to be; we can do this in
  // O(lg(n)) function generations rather than O(n)
  if constexpr (dim == 2) {
    return new_e;
  } else {
    return rosenbrock_function_helper<dim - 2>(new_e, alpha, var_id + 2);
  }
}

template <size_t dim> constexpr auto rosenbrock_function(double alpha = 1.0) {
  static_assert(
      dim >= 2,
      "The Rosenbrock function isn't defined for less than 2 dimensions");
  static_assert(
      dim % 2 == 0,
      "The Rosenbrock function isn't defined for an odd number of dimensions");
  const auto_diff::variable<double> x0(0), x1(1);
  const auto e = rosenbrock_function_term(alpha, x0, x1);
  if constexpr (dim == 2) {
    return e;
  } else {
    return rosenbrock_function_helper<dim - 2>(e, alpha, 2);
  }
}

template <typename optimizer>
const std::vector<PetscScalar> &
fail_solve_throw(optimizer &opt, const std::vector<PetscScalar> &starting_pt) {
  EXPECT_NO_THROW(return opt.solve(starting_pt));
  throw std::runtime_error("Error minimizing the Rosenbrock function");
}

template <size_t dim> void test_optimizer() {
  constexpr auto f = rosenbrock_function<dim>(0.25);
  auto_diff::optimize_tao::NL_Smooth_Optimizer<decltype(f)> opt(f, dim);

  std::random_device rd;
  rngAlg engine(rd());
  std::uniform_real_distribution<double> pdf(-16.0, 16.0);

  std::vector<PetscScalar> starting_pt(dim);
  for (size_t i = 0; i < dim; ++i) {
    starting_pt[i] = pdf(engine);
  }

  const std::vector<PetscScalar> &arg_min = fail_solve_throw(opt, starting_pt);
  for (size_t i = 0; i < arg_min.size(); ++i) {
    // Verify we're actually near an extremum
    EXPECT_NEAR(f.deriv(i).eval(arg_min), 0.0, 1e-8);
  }
}

TEST(nonlinear_optimizer, rosenbrock_function2) { test_optimizer<2>(); }
TEST(nonlinear_optimizer, rosenbrock_function4) { test_optimizer<4>(); }
TEST(nonlinear_optimizer, rosenbrock_function16) { test_optimizer<16>(); }

int main(int argc, char **argv) {
  petsc_helpers::PetscScope p;
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
