
#include <cmath>

#include "autodiff.hpp"
#include "autodiff_optimize.hpp"
#include "autodiff_transcendental.hpp"

#include <random>

#include <gtest/gtest.h>

using namespace auto_diff;

using rngAlg = std::mt19937_64;

template <typename expr_t>
void finitediff_test(const expr_t &e, const variable<double> v, const double xc,
                     const std::string testname) {
  constexpr double eps = std::numeric_limits<double>::epsilon();
  const double x0 = xc * (1.0 - 1e-6);
  const double x1 = xc * (1.0 + 1e-6);
  const double delta = x1 - x0;
  const double fdiff = (e.eval(v, x1) - e.eval(v, x0)) / delta;
  const double imdiff = e.deriv(v.id()).eval(v, xc);
  // We want to ensure the relative difference isn't too large, while also
  // ensuring the minimum relative distance isn't too small as occurs when the
  // function is nearly flat
  // ie, finite difference on cos(x) has large relative error when x is near 0;
  // tests have shown the absolute error to be as high as 9e-5 when the relative
  // error is nearly unbounded (fdiff is essentially 0)
  EXPECT_NEAR(imdiff, fdiff, std::abs(fdiff) * eps * 1e12 + 1e-4)
      << "Failed: " << testname << " at " << xc << "\n";
}

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

TEST(exp_eval, autodiff) {
  constexpr variable<double> s(1), t(2), u(3), v(4);
  constexpr auto stuv = sqrt(s) + cbrt(t) + exp(u) + log(v + 1.0);

  EXPECT_EQ(stuv.eval(s.id(), 4.0), 2.0 + 1.0);
  EXPECT_EQ(stuv.eval(t.id(), 27.0), 3.0 + 1.0);
  EXPECT_NEAR(stuv.eval(u.id(), 3.0), M_E * M_E * M_E, 5e-15);
  EXPECT_EQ(stuv.eval(v.id(), M_E * M_E * M_E * M_E - 1.0), 4.0 + 1.0);
  EXPECT_EQ(stuv.eval(std::pair<auto_diff::id_t, double>{t.id(), 5.0},
                      std::pair<auto_diff::id_t, double>{u.id(), 4.0},
                      std::pair<auto_diff::id_t, double>{v.id(), 11.0},
                      std::pair<auto_diff::id_t, double>{s.id(), 16.0}),
            sqrt(16.0) + cbrt(5.0) + exp(4.0) + log(11.0 + 1.0));

  std::random_device rd;
  rngAlg engine(rd());
  std::uniform_real_distribution<double> pdf(0.0, 128.0);
  for (int i = 0; i < 1000; ++i) {
    const double rval = pdf(engine);
    EXPECT_NEAR(stuv.eval(s.id(), rval), sqrt(rval) + 1.0, 5e-16);
    EXPECT_NEAR(stuv.eval(t.id(), rval), cbrt(rval) + 1.0, 5e-16);
    EXPECT_NEAR(stuv.eval(u.id(), rval / 8.0), exp(rval / 8.0), 5e-16);
    EXPECT_NEAR(stuv.eval(v.id(), rval), log(rval + 1.0) + 1.0, 5e-16);

    EXPECT_EQ(sqrt(4.0 * s).deriv(s.id()).eval(s, rval), 1.0 / sqrt(rval));
    finitediff_test(sqrt(4.0 * s), s, rval, "Sqrt");
    EXPECT_NEAR(cbrt(12.0 * s).deriv(s.id()).eval(s, rval),
                4.0 / (cbrt(12.0 * rval) * cbrt(12.0 * rval)), 5e-13);
    // Test error to deal with
    // The difference between cbrt(12.0 * s).deriv(s.id()).eval(s, rval) and 4.0
    // / (cbrt(12.0 * rval) * cbrt(12.0 * rval)) is 1.8189894035458565e-12,
    // which exceeds 5e-13, where
    // cbrt(12.0 * s).deriv(s.id()).eval(s, rval) evaluates to
    // 8371.7685990041518,
    // 4.0 / (cbrt(12.0 * rval) * cbrt(12.0 * rval)) evaluates to
    // 8371.76859900415, and
    // 5e-13 evaluates to 4.9999999999999999e-13.
    finitediff_test(cbrt(4.0 * s), s, rval, "Cbrt");
    EXPECT_EQ(exp(4.0 * s).deriv(s.id()).eval(s, rval), 4.0 * exp(4.0 * rval));
    finitediff_test(exp(4.0 * s), s, rval, "Exp");
    EXPECT_EQ(log(4.0 * s + 1.0).deriv(s.id()).eval(s, rval),
              4.0 / (4.0 * rval + 1.0));
    finitediff_test(log(4.0 * s + 1.0), s, rval, "Log");
  }
}

TEST(trig_eval, autodiff) {
  constexpr variable<double> s(1), t(2);
  constexpr auto st = sin(t) + sin(s);
  EXPECT_EQ(st.eval(t.id(), 0.0), 0.0);
  EXPECT_EQ(st.eval(t.id(), M_PI / 2.0), 1.0);
  EXPECT_NEAR(st.eval(t.id(), M_PI), 0.0, 2e-16);
  EXPECT_NEAR(st.eval(std::pair<auto_diff::id_t, double>{t.id(), M_PI},
                      std::pair<auto_diff::id_t, double>{s.id(), M_PI}),
              0.0, 4e-16);
  EXPECT_NEAR(st.eval(t.id(), M_PI, s.id(), M_PI), 0.0, 4e-16);
  const Cos<variable<double>> ct = cos(t);
  EXPECT_EQ(ct.eval(t.id(), 0.0), 1.0);
  EXPECT_NEAR(ct.eval(t.id(), M_PI / 2.0), 0.0, 1e-16);
  EXPECT_NEAR(ct.eval(t.id(), M_PI), -1.0, 1e-16);
  const Tan<variable<double>> tt = tan(t);
  EXPECT_EQ(tt.eval(t.id(), 0.0), 0.0);
  EXPECT_NEAR(tt.eval(t.id(), M_PI), 0.0, 2e-16);

  std::random_device rd;
  rngAlg engine(rd());
  std::uniform_real_distribution<double> pdf(-7.0, 7.0);
  for (int i = 0; i < 1000; ++i) {
    const double rval = [&pdf, &engine]() {
      // We need to make certain that rval isn't too close to pi/2 or 3pi/2 for
      // the finite difference test with tan
      const double v = pdf(engine);
      for (auto check : {M_PI / 2.0, 3.0 * M_PI / 2.0}) {
        if ((std::abs(std::abs(v) - check)) < 1e-2) {
          return v + std::copysign(1.0, std::abs(v) - check) *
                         std::copysign(1.0, v) * 1e-2;
        }
      }
      return v;
    }();
    EXPECT_EQ(st.eval(t, rval), std::sin(rval));
    finitediff_test(st, t, rval, "Sin");
    EXPECT_EQ(st.deriv(t.id()).eval(t, rval), std::cos(rval));
    EXPECT_EQ(ct.eval(t, rval), std::cos(rval));
    finitediff_test(ct, t, rval, "Cos");
    EXPECT_EQ(ct.deriv(t.id()).eval(t, rval), -std::sin(rval));
    EXPECT_EQ(tt.eval(t, rval), std::tan(rval));
    finitediff_test(tt, t, rval, "Tan");
    EXPECT_EQ(tt.deriv(t.id()).eval(t, rval),
              1.0 / (std::cos(rval) * std::cos(rval)));
  }
}

TEST(invtrig_eval, autodiff) {
  constexpr variable<double> s(1);
  EXPECT_EQ(asin(s).eval(s, 0.0), 0.0);
  EXPECT_EQ(asin(s).eval(s, 1.0), M_PI / 2.0);
  EXPECT_EQ(acos(s).eval(s, 0.0), M_PI / 2.0);
  EXPECT_EQ(acos(s).eval(s, 1.0), 0.0);
  EXPECT_EQ(atan(s).eval(s, 0.0), 0.0);
  EXPECT_EQ(atan(s).eval(s, 1.0), M_PI / 4.0);

  std::random_device rd;
  rngAlg engine(rd());
  std::uniform_real_distribution<double> pdf(-1.0 + 0.06125, 1.0 - 0.06125);
  for (int i = 0; i < 1000; ++i) {
    const double rval = pdf(engine);
    EXPECT_EQ(asin(s).eval(s, rval), std::asin(rval));
    EXPECT_EQ(asin(s).deriv(s.id()).eval(s, rval),
              1.0 / std::sqrt(1.0 - rval * rval));
    finitediff_test(asin(s), s, rval, "ASin");

    EXPECT_EQ(acos(s).eval(s, rval), std::acos(rval));
    EXPECT_EQ(acos(s).deriv(s.id()).eval(s, rval),
              -1.0 / std::sqrt(1.0 - rval * rval));
    finitediff_test(acos(s), s, rval, "ACos");

    EXPECT_EQ(atan(s).eval(s, rval), std::atan(rval));
    EXPECT_EQ(atan(s).deriv(s.id()).eval(s, rval), 1.0 / (1.0 + rval * rval));
    finitediff_test(atan(s), s, rval, "ATan");
  }
}

TEST(pow_eval, autodiff) {
  constexpr variable<double> s(1), t(2);
  // Sanity checks for the three exponential forms
  constexpr auto p1 = pow(s, t);
  EXPECT_EQ(p1.eval(s, 1.0), 1.0);
  EXPECT_EQ(p1.eval(s, 1.0, 2, 4.0), 1.0);
  EXPECT_EQ(p1.eval(s, 2.0, 2, 4.0), 16.0);
  EXPECT_EQ(p1.deriv(s.id()).eval(s, 2.0, t, 4.0), 4.0 * 2.0 * 2.0 * 2.0);

  constexpr auto p2 = pow(s, 3.0);
  EXPECT_EQ(p2.eval(s, 1.0), 1.0);
  EXPECT_EQ(p2.eval(s, 2.0), 8.0);
  EXPECT_EQ(p2.eval(s, -3.0), -27.0);
  EXPECT_EQ(p2.deriv(s.id()).eval(s, -3.0), 27.0);

  constexpr auto p3 = pow(3.0, t);
  EXPECT_EQ(p3.eval(t, 1.0), 3.0);
  EXPECT_EQ(p3.eval(t, 2.0), 9.0);
  EXPECT_EQ(p3.eval(t, -3.0), 1.0 / 27.0);
  EXPECT_EQ(p3.deriv(t.id()).eval(t, -3.0),
            std::log(3.0) * std::pow(3.0, -3.0));

  std::random_device rd;
  rngAlg engine(rd());
  std::uniform_real_distribution<double> pdf(0.001, 8.0);
  for (int i = 0; i < 1000; ++i) {
    const double base = pdf(engine);
    const double exponent = pdf(engine);
    EXPECT_EQ(p1.eval(s, base, t, exponent), std::pow(base, exponent));
    EXPECT_EQ(p1.deriv(s.id()).eval(s, base, t, exponent),
              std::pow(base, exponent - 1.0));
  }
}

TEST(polynomial_grad, autodiff) {
  constexpr variable<double> x(1), y(2);
  constexpr double c = 432.043241;
  constexpr auto e1 = x + c;
  constexpr auto e1_dx = 1.0 + 0.0;
  std::map<auto_diff::id_t, std::remove_const<decltype(e1_dx)>::type> grad1 =
      gradient(e1);
  EXPECT_EQ(grad1.size(), 1);
  EXPECT_EQ(e1_dx, grad1[x.id()]);

  constexpr auto e2 = x + 2.0 * y + c;
  constexpr auto e2_dx = 1.0 + 2.0 * 0.0 + 0.0;
  constexpr auto e2_dy = 0.0 + 2.0 * 1.0 + 0.0;
  std::map<auto_diff::id_t, std::remove_const<decltype(e2_dx)>::type> grad2 =
      gradient(e2);
  EXPECT_EQ(grad2.size(), 2);
  EXPECT_EQ(e2_dx, grad2[x.id()]);
  EXPECT_EQ(e2_dy, grad2[y.id()]);

  constexpr auto e3 = x * x + 5.0 * x + c;
  constexpr auto e3_dx = 1.0 * x + x * 1.0 + 5.0 * 1.0;
  constexpr auto e3_dxdx = 1.0 * 1.0 + 1.0 * 1.0;
  std::map<auto_diff::id_t, std::remove_const<decltype(e3_dx)>::type> grad3 =
      gradient(e3);
  std::map<std::pair<auto_diff::id_t, auto_diff::id_t>,
           std::remove_const<decltype(e3_dxdx)>::type>
      hessian3 = hessian(e3);
  EXPECT_EQ(hessian3.size(), 1 * 1);
  EXPECT_EQ(e3_dxdx, hessian3.at({x.id(), x.id()}));

  constexpr auto e4 =
      x * x + 5.0 * x * y - 2.0 * y * y + M_PI * x + M_E * y + c;
  constexpr auto e4_dx = (1.0 * x + x * 1.0) + (5.0 * 1.0 * y + 5.0 * x * 0.0) -
                         (2.0 * 0.0 * y + 2.0 * y * 0.0) + M_PI * 1.0 +
                         M_E * 0.0;
  constexpr auto e4_dy = (0.0 * x + x * 0.0) + (5.0 * 0.0 * y + 5.0 * x * 1.0) -
                         (2.0 * 1.0 * y + 2.0 * y * 1.0) + M_PI * 0.0 +
                         M_E * 1.0;
  constexpr auto e4_dxdx = (1.0 * 1.0 + 1.0 * 1.0) +
                           (5.0 * 1.0 * 0.0 + 5.0 * 1.0 * 0.0) -
                           (2.0 * 0.0 * 0.0 + 2.0 * 0.0 * 0.0);
  constexpr auto e4_dxdy = (0.0 * 1.0 + 1.0 * 0.0) +
                           (5.0 * 0.0 * 0.0 + 5.0 * 1.0 * 1.0) -
                           (2.0 * 1.0 * 0.0 + 2.0 * 0.0 * 1.0);
  constexpr auto e4_dydy = (0.0 * 0.0 + 0.0 * 0.0) +
                           (5.0 * 0.0 * 1.0 + 5.0 * 0.0 * 1.0) -
                           (2.0 * 1.0 * 1.0 + 2.0 * 1.0 * 1.0);
  const std::map<auto_diff::id_t, std::remove_const<decltype(e4_dx)>::type>
      grad4 = gradient(e4);
  const std::map<std::pair<auto_diff::id_t, auto_diff::id_t>,
                 std::remove_const<decltype(e4_dxdx)>::type>
      hessian4 = hessian(e4);
  EXPECT_EQ(hessian4.size(), 2 * 2);
  EXPECT_EQ(e4_dxdx, hessian4.at({x.id(), x.id()}));
  EXPECT_EQ(e4_dxdy, hessian4.at({x.id(), y.id()}));
  EXPECT_EQ(e4_dxdy, hessian4.at({y.id(), x.id()}));
  EXPECT_EQ(e4_dydy, hessian4.at({y.id(), y.id()}));

  std::random_device rd;
  rngAlg engine(rd());
  std::uniform_real_distribution<double> pdf(-1024.0, 1024.0);
  for (int i = 0; i < 100; i++) {
    const double x_eval = pdf(engine);
    const double y_eval = pdf(engine);
    EXPECT_EQ(e4_dx.eval(x.id(), x_eval, y.id(), y_eval),
              grad4.at(x.id()).eval(x.id(), x_eval, y.id(), y_eval));
    EXPECT_EQ(e4_dy.eval(x.id(), x_eval, y.id(), y_eval),
              grad4.at(y.id()).eval(x.id(), x_eval, y.id(), y_eval));
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
