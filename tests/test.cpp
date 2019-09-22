
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

  const variable<double> x(1);
  EXPECT_EQ(x.eval(x.id(), 4.0), 4.0);

  const multiplication<variable<double>, variable<double>> xsqr = x * x;
  for (int i = 0; i < 1000; ++i) {
    const double rval = pdf(engine);
    EXPECT_EQ(x.eval(x.id(), rval), rval);
    const double e2 = xsqr.eval(x.id(), rval);
    EXPECT_EQ(e2, rval * rval);

    constexpr double c = -20.0;
    addition<
        addition<addition<variable<double>,
                          multiplication<variable<double>, variable<double>>>,
                 double>,
        variable<double>>
        x_quadratic = x + x * x + c + x;
    const double e4 = x_quadratic.eval(x.id(), rval);
    EXPECT_EQ(e4, rval + rval * rval + c + rval);
    const double e5 = x_quadratic.eval(x.id(), 0.5);
    EXPECT_EQ(e5, 0.75 + c + 0.5);
    EXPECT_EQ(x_quadratic.deriv().eval(x.deriv_id(0), 1.0), 0.0);
    EXPECT_EQ(x_quadratic.deriv().eval(x.deriv_id(1), 1.0), 2.0);
  }
}

TEST(trig_eval, autodiff) {
  const variable<double> t(1);
  const Sin<variable<double>> st = Sin(t);
  EXPECT_EQ(st.eval(t.id(), 0.0), 0.0);
  EXPECT_EQ(st.eval(t.id(), M_PI / 2.0), 1.0);
  EXPECT_NEAR(st.eval(t.id(), M_PI), 0.0, 2e-16);
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
