
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

template <typename expr_t, typename enable_ = std::enable_if_t<
                               std::is_base_of_v<expr, expr_t>, void>>
static constexpr Sqrt<expr_t> sqrt(expr_t e) {
  return Sqrt<expr_t>(e);
};
template <typename expr_t, typename enable_ = std::enable_if_t<
                               std::is_base_of_v<expr, expr_t>, void>>
static constexpr Cbrt<expr_t> cbrt(expr_t e) {
  return Cbrt<expr_t>(e);
};
template <typename expr_t, typename enable_ = std::enable_if_t<
                               std::is_base_of_v<expr, expr_t>, void>>
static constexpr Exp<expr_t> exp(expr_t e) {
  return Exp<expr_t>(e);
};
template <typename expr_t, typename enable_ = std::enable_if_t<
                               std::is_base_of_v<expr, expr_t>, void>>
static constexpr Log<expr_t> log(expr_t e) {
  return Log<expr_t>(e);
};

template <typename expr_t_> class Sin;
template <typename expr_t_> class Cos;
template <typename expr_t_> class Tan;

template <typename expr_t, typename enable_ = std::enable_if_t<
                               std::is_base_of_v<expr, expr_t>, void>>
static constexpr Sin<expr_t> sin(expr_t e) {
  return Sin(e);
};
template <typename expr_t, typename enable_ = std::enable_if_t<
                               std::is_base_of_v<expr, expr_t>, void>>
static constexpr Cos<expr_t> cos(expr_t e) {
  return Cos(e);
};
template <typename expr_t, typename enable_ = std::enable_if_t<
                               std::is_base_of_v<expr, expr_t>, void>>
static constexpr Tan<expr_t> tan(expr_t e) {
  return Tan(e);
};

template <typename expr_t_> class Asin;
template <typename expr_t_> class Acos;
template <typename expr_t_> class Atan;

template <typename expr_t, typename enable_ = std::enable_if_t<
                               std::is_base_of_v<expr, expr_t>, void>>
static constexpr Asin<expr_t> asin(expr_t e) {
  return Asin<expr_t>(e);
};
template <typename expr_t, typename enable_ = std::enable_if_t<
                               std::is_base_of_v<expr, expr_t>, void>>
static constexpr Acos<expr_t> acos(expr_t e) {
  return Acos<expr_t>(e);
};
template <typename expr_t, typename enable_ = std::enable_if_t<
                               std::is_base_of_v<expr, expr_t>, void>>
static constexpr Atan<expr_t> atan(expr_t e) {
  return Atan<expr_t>(e);
};

template <typename lhs_expr_t_, typename rhs_expr_t_> class Pow;
template <
    typename lhs_expr_t, typename rhs_expr_t,
    typename enable_ = std::enable_if_t<std::is_base_of_v<expr, lhs_expr_t> ||
                                            std::is_base_of_v<expr, rhs_expr_t>,
                                        void>>
static constexpr Pow<lhs_expr_t, rhs_expr_t> pow(lhs_expr_t lhs,
                                                 rhs_expr_t rhs) {
  return Pow<lhs_expr_t, rhs_expr_t>(lhs, rhs);
};

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

  constexpr auto deriv(const id_t deriv_id) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return this->val_.deriv(deriv_id) / (space(2) * *this);
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

  constexpr auto deriv(const id_t deriv_id) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return this->val_.deriv(deriv_id) / (space(3) * *this * *this);
    } else {
      return space(0);
    }
  }
};

template <typename expr_t_>
class Exp
    : public unary_op<expr_t_,
                      transcendental::exp_function<expr_domain<expr_t_>>> {
public:
  using uop =
      unary_op<expr_t_, transcendental::exp_function<expr_domain<expr_t_>>>;
  using expr_t = typename uop::expr_t;
  using space = typename uop::space;

  constexpr explicit Exp(expr_t val) : uop(val) {}

  constexpr auto deriv(const id_t deriv_id) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return *this * this->val_.deriv(deriv_id);
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

  constexpr auto deriv(const id_t deriv_id) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return this->val_.deriv(deriv_id) / this->val_;
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

  constexpr auto deriv(const id_t deriv_id) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return Cos(this->val_) * this->val_.deriv(deriv_id);
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

  constexpr auto deriv(const id_t deriv_id) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      // Clang weirdness requires the leading space(0)
      return space(0) - Sin(this->val_) * this->val_.deriv(deriv_id);
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

  constexpr auto deriv(const id_t deriv_id) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return this->val_.deriv(deriv_id) / (cos(this->val_) * cos(this->val_));
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

  constexpr auto deriv(const id_t deriv_id) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return this->val_.deriv(deriv_id) /
             sqrt(space(1.0) - this->val_ * this->val_);
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

  constexpr auto deriv(const id_t deriv_id) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return -(this->val_.deriv(deriv_id) /
							 sqrt(space(1.0) - this->val_ * this->val_));
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

  constexpr auto deriv(const id_t deriv_id) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return this->val_.deriv(deriv_id) /
             (this->val_ * this->val_ + space(1.0));
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
  using lhs_expr_t = typename bop::lhs_expr_t;
  using rhs_expr_t = typename bop::rhs_expr_t;
  using space = typename bop::space;

  constexpr explicit Pow(lhs_expr_t base, rhs_expr_t exp) : bop(base, exp) {}

  constexpr auto deriv(const id_t deriv_id) const {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return Log(this->lhs) * Pow(this->lhs, this->rhs) *
                   this->rhs.deriv(deriv_id) +
               Pow(this->lhs, this->rhs - 1) * this->rhs *
                   this->lhs.deriv(deriv_id);
      } else {
        return Pow(this->lhs, this->rhs - 1) * this->rhs *
               this->lhs.deriv(deriv_id);
      }
    } else if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
      return Log(this->lhs) * Pow(this->lhs, this->rhs) *
             this->rhs.deriv(deriv_id);
    } else {
      // Should never happen
      return space(0);
    }
  }
};

} // namespace auto_diff

#endif // AUTODIFF_TRANSCENDENTAL_HPP
