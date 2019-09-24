
#ifndef AUTODIFF_TRANSCENDENTAL_HPP
#define AUTODIFF_TRANSCENDENTAL_HPP

#include "autodiff.hpp"

#include <cmath>
#include <functional>

namespace auto_diff {

namespace transcendental {

template <typename space> struct sqrt_function {
  constexpr space operator()(const space &x) const { return std::sqrt(x); }
};

template <typename space> struct cbrt_function {
  constexpr space operator()(const space &x) const { return std::cbrt(x); }
};

template <typename space> struct exp_function {
  constexpr space operator()(const space &x) const { return std::exp(x); }
};

template <typename space> struct log_function {
  constexpr space operator()(const space &x) const { return std::log(x); }
};

template <typename space> struct sin_function {
  constexpr space operator()(const space &x) const { return std::sin(x); }
};

template <typename space> struct cos_function {
  constexpr space operator()(const space &x) const { return std::cos(x); }
};

template <typename space> struct tan_function {
  constexpr space operator()(const space &x) const { return std::tan(x); }
};

template <typename space> struct asin_function {
  constexpr space operator()(const space &x) const { return std::asin(x); }
};

template <typename space> struct acos_function {
  constexpr space operator()(const space &x) const { return std::acos(x); }
};

template <typename space> struct atan_function {
  constexpr space operator()(const space &x) const { return std::atan(x); }
};

template <typename space> struct pow_function {
  constexpr space operator()(const space &x, const space &y) const {
    return std::pow(x, y);
  }
};

} // namespace transcendental

template <typename expr_t_> class Sqrt;
template <typename expr_t_> class Cbrt;
template <typename expr_t_> class Exp;
template <typename expr_t_> class Log;

template <typename expr_t_> class Sin;
template <typename expr_t_> class Cos;
template <typename expr_t_> class Tan;

template <typename expr_t_> class Asin;
template <typename expr_t_> class Acos;
template <typename expr_t_> class Atan;

template <typename lhs_expr_t_, typename rhs_expr_t_> class Pow;

// Roots, expoonents, logarithms

template <typename expr_t_>
class Sqrt
    : public unary_op<expr_t_,
                      transcendental::sqrt_function<expr_domain<expr_t_>>> {
public:
  using uop =
      unary_op<expr_t_, transcendental::sqrt_function<expr_domain<expr_t_>>>;
  using expr_t = typename uop::expr_t;
  using space = typename uop::space;

  constexpr explicit Sqrt(expr_t val) : uop(val) {}

  constexpr auto deriv() const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return this->val_.deriv() / (space(2) * *this);
    } else {
      return space(0);
    }
  }
};

template <typename expr_t_>
class Cbrt
    : public unary_op<expr_t_,
                      transcendental::cbrt_function<expr_domain<expr_t_>>> {
public:
  using uop =
      unary_op<expr_t_, transcendental::cbrt_function<expr_domain<expr_t_>>>;
  using expr_t = typename uop::expr_t;
  using space = typename uop::space;

  constexpr explicit Cbrt(expr_t val) : uop(val) {}

  constexpr auto deriv() const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return this->val_.deriv() / (space(3) * Pow(*this, space(2)));
    } else {
      return space(0);
    }
  }
};

template <typename expr_t_>
class Exp : public unary_op<expr_t_, typename expr_domain<expr_t_>::space> {
public:
  using uop = unary_op<expr_t_, typename expr_domain<expr_t_>::space>;
  using expr_t = typename uop::expr_t;
  using space = typename uop::space;

  constexpr explicit Exp(expr_t val) : uop(val) {}

  constexpr auto deriv() const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return *this * this->val_.deriv();
    } else {
      return space(0);
    }
  }
};

template <typename expr_t_>
class Log
    : public unary_op<expr_t_,
                      transcendental::log_function<expr_domain<expr_t_>>> {
public:
  using uop =
      unary_op<expr_t_, transcendental::log_function<expr_domain<expr_t_>>>;
  using expr_t = typename uop::expr_t;
  using space = typename uop::space;

  constexpr explicit Log(expr_t val) : uop(val) {}

  constexpr auto deriv() const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return this->val_.deriv() / this->val_;
    } else {
      return space(0);
    }
  }
};

// Trigonometric functions

template <typename expr_t_>
class Sin
    : public unary_op<expr_t_,
                      transcendental::sin_function<expr_domain<expr_t_>>> {
public:
  using uop =
      unary_op<expr_t_, transcendental::sin_function<expr_domain<expr_t_>>>;
  using expr_t = typename uop::expr_t;
  using space = typename uop::space;

  constexpr explicit Sin(expr_t_ val) : uop(val) {}

  constexpr auto deriv() const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return Cos(this->val_) * this->val_.deriv();
    } else {
      return space(0);
    }
  }
};

template <typename expr_t_>
class Cos
    : public unary_op<expr_t_,
                      transcendental::cos_function<expr_domain<expr_t_>>> {
public:
  using uop =
      unary_op<expr_t_, transcendental::cos_function<expr_domain<expr_t_>>>;
  using expr_t = typename uop::expr_t;
  using space = typename uop::space;

  constexpr explicit Cos(expr_t_ val) : uop(val) {}

  constexpr auto deriv() const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      // Clang weirdness requires the leading space(0)
      return space(0) - Sin(this->val_) * this->val_.deriv();
    } else {
      return space(0);
    }
  }
};

template <typename expr_t_>
class Tan
    : public unary_op<expr_t_,
                      transcendental::tan_function<expr_domain<expr_t_>>> {
public:
  using uop =
      unary_op<expr_t_, transcendental::tan_function<expr_domain<expr_t_>>>;
  using expr_t = typename uop::expr_t;
  using space = typename uop::space;

  constexpr explicit Tan(expr_t_ val) : uop(val) {}

  constexpr auto deriv() const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return (Pow<Tan<expr_t>, space>(Tan(this->val_), space(2)) + 1) *
             this->val_.deriv();
    } else {
      return space(0);
    }
  }
};

// Inverse trigonometric functions

template <typename expr_t_>
class Asin
    : public unary_op<expr_t_,
                      transcendental::asin_function<expr_domain<expr_t_>>> {
public:
  using uop =
      unary_op<expr_t_, transcendental::asin_function<expr_domain<expr_t_>>>;
  using expr_t = typename uop::expr_t;
  using space = typename uop::space;

  constexpr explicit Asin(expr_t_ val) : uop(val) {}

  constexpr auto deriv() const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return -1 / sqrt(1 - pow(this->val_, space(2)));
    } else {
      return space(0);
    }
  }
};

template <typename expr_t_>
class Acos
    : public unary_op<expr_t_,
                      transcendental::acos_function<expr_domain<expr_t_>>> {
public:
  using uop =
      unary_op<expr_t_, transcendental::acos_function<expr_domain<expr_t_>>>;
  using expr_t = typename uop::expr_t;
  using space = typename uop::space;

  constexpr explicit Acos(expr_t val) : uop(val) {}

  constexpr auto deriv() const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return 1 / sqrt(1 - pow(this->val_, space(2)));
    } else {
      return space(0);
    }
  }
};

template <typename expr_t_>
class Atan
    : public unary_op<expr_t_,
                      transcendental::atan_function<expr_domain<expr_t_>>> {
public:
  using uop =
      unary_op<expr_t_, transcendental::atan_function<expr_domain<expr_t_>>>;
  using expr_t = typename uop::expr_t;
  using space = typename uop::space;

  constexpr explicit Atan(expr_t val) : uop(val) {}

  constexpr auto deriv() const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return 1 / (Pow(this->val_, space(2)) + 1);
    } else {
      return space(0);
    }
  }
};

// x ^ y
template <typename lhs_expr_t_, typename rhs_expr_t_>
class Pow
    : public binary_op<lhs_expr_t_, rhs_expr_t_,
                       transcendental::pow_function<expr_domain<lhs_expr_t_>>> {
public:
  using bop = binary_op<lhs_expr_t_, rhs_expr_t_,
                        transcendental::pow_function<expr_domain<lhs_expr_t_>>>;
  using expr_t = typename bop::expr_t;
  using space = typename bop::space;

  constexpr explicit Pow(expr_t val) : bop(val) {}

  constexpr auto deriv() const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return Pow(this->lhs, this->rhs) * this->lhs.deriv() +
             Pow(this->lhs, this->rhs - 1) * this->lhs.deriv();
    } else {
      return space(0);
    }
  }
};

} // namespace auto_diff

#endif // AUTODIFF_TRANSCENDENTAL_HPP
