
#include <cmath>

#include "autodiff.hpp"
#include "autodiff_transcendental.hpp"

#include <random>

#include <gtest/gtest.h>

using namespace auto_diff;

using rngAlg = std::mt19937_64;

TEST(polynomial_eval, autodiff) {
  std::random_device rd;
  rngAlg engine(rd());
  std::uniform_real_distribution<double> pdf(-1024.0, 1024.0);

  constexpr variable<double> x(1), y(2);
  EXPECT_EQ(x.eval(x.id(), 4.0), 4.0);
  EXPECT_EQ(x.eval(std::pair{x.id(), 4.0}, std::pair{x.id() + 1, 10.0},
                   std::pair{x.id(), -10.0}, std::pair{x.id() + 53.0, 100.0}),
            4.0);

  EXPECT_EQ(x.deriv(x.id()), 1.0);
  (x + x).deriv(x.id());
  double v = (x + y).eval(std::pair<auto_diff::id_t, double>{x.id(), 8.0},
                          std::pair<auto_diff::id_t, double>{y.id(), 4.0});
  EXPECT_EQ(v, 12.0);

  constexpr double c = -20.0;
  constexpr addition<
      subtraction<addition<variable<double>,
                           multiplication<variable<double>, variable<double>>>,
                  double>,
      variable<double>>
      x_quadratic = x + x * x - c + x;
  const multiplication<variable<double>, variable<double>> xsqr = x * x;
  for (int i = 0; i < 1000; ++i) {
    const double rval = pdf(engine);
    EXPECT_EQ(x.eval(x.id(), rval), rval);
    const double e2 = xsqr.eval(x.id(), rval);
    EXPECT_EQ(e2, rval * rval);

    const double e4 = x_quadratic.eval(x.id(), rval);
    EXPECT_EQ(e4, rval + rval * rval - c + rval);
  }
  const double e5 = x_quadratic.eval(x.id(), 0.5);
  EXPECT_EQ(e5, 0.75 - c + 0.5);
  EXPECT_EQ(x_quadratic.deriv(x.id()).eval(x.id(), 1.0), 4.0);
}

TEST(trig_eval, autodiff) {
  constexpr variable<double> s(1), t(2);
  constexpr auto st = Sin(t) + Sin(s);
  EXPECT_EQ(st.eval(t.id(), 0.0), 0.0);
  EXPECT_EQ(st.eval(t.id(), M_PI / 2.0), 1.0);
  EXPECT_NEAR(st.eval(t.id(), M_PI), 0.0, 2e-16);
  EXPECT_NEAR(st.eval(std::pair<auto_diff::id_t, double>{t.id(), M_PI},
                      std::pair<auto_diff::id_t, double>{s.id(), M_PI}),
              0.0, 4e-16);
  EXPECT_NEAR(st.eval(t.id(), M_PI, s.id(), M_PI), 0.0, 4e-16);
  const Cos<variable<double>> ct = Cos(t);
  EXPECT_EQ(ct.eval(t.id(), 0.0), 1.0);
  EXPECT_NEAR(ct.eval(t.id(), M_PI / 2.0), 0.0, 1e-16);
  EXPECT_NEAR(ct.eval(t.id(), M_PI), -1.0, 1e-16);
  const Tan<variable<double>> tt = Tan(t);
  EXPECT_EQ(tt.eval(t.id(), 0.0), 0.0);
  EXPECT_NEAR(tt.eval(t.id(), M_PI), 0.0, 2e-16);

  std::random_device rd;
  rngAlg engine(rd());
  std::uniform_real_distribution<double> pdf(-1024.0, 1024.0);
  for (int i = 0; i < 1000; ++i) {
    const double rval = pdf(engine);
    EXPECT_EQ(st.eval(t.id(), rval), std::sin(rval));
    EXPECT_EQ(ct.eval(t.id(), rval), std::cos(rval));
    EXPECT_EQ(tt.eval(t.id(), rval), std::tan(rval));
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
