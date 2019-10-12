
#include <cmath>

#include "autodiff.hpp"
#include "autodiff_optimize.hpp"
#include "autodiff_transcendental.hpp"

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

int main(int argc, char **argv) {
  PetscInitializeNoArguments();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
