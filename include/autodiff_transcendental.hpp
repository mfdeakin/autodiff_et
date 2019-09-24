
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

  constexpr auto subs(const id_t eval_id, const space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return Exp(this->val_.subs(eval_id, v));
    } else {
      return exp(this->val_);
    }
  }

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

  constexpr space eval(const id_t eval_id, space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return log(this->val_.eval(eval_id, v));
    } else {
      return log(this->val_);
    }
  }

  constexpr space eval(const std::vector<space> &values) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return log(this->val_.eval(values));
    } else {
      return log(this->val_);
    }
  }

  constexpr auto subs(const id_t eval_id, const space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return Log(this->val_.subs(eval_id, v));
    } else {
      return log(this->val_);
    }
  }

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

  constexpr space eval(const id_t eval_id, space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return sin(this->val_.eval(eval_id, v));
    } else {
      return sin(this->val_);
    }
  }

  constexpr space eval(const std::vector<space> &values) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return sin(this->val_.eval(values));
    } else {
      return sin(this->val_);
    }
  }

  constexpr auto subs(const id_t eval_id, const space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return Sin(this->val_.subs(eval_id, v));
    } else {
      return Sin(this->val_);
    }
  }

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

  constexpr space eval(const id_t eval_id, space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return cos(this->val_.eval(eval_id, v));
    } else {
      return cos(this->val_);
    }
  }

  constexpr space eval(const std::vector<space> &values) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return cos(this->val_.eval(values));
    } else {
      return cos(this->val_);
    }
  }

  constexpr auto subs(const id_t eval_id, const space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return Cos(this->val_.subs(eval_id, v));
    } else {
      return cos(this->val_);
    }
  }

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

  constexpr space eval(const id_t eval_id, space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return tan(this->val_.eval(eval_id, v));
    } else {
      return tan(this->val_);
    }
  }

  constexpr space eval(const std::vector<space> &values) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return tan(this->val_.eval(values));
    } else {
      return tan(this->val_);
    }
  }

  constexpr auto subs(const id_t eval_id, const space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return Tan(this->val_.subs(eval_id, v));
    } else {
      return tan(this->val_);
    }
  }

  constexpr auto deriv() const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return (Pow(Tan(this->val_), space(2)) + 1) * this->val_.deriv();
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

  constexpr space eval(const id_t eval_id, space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return asin(this->val_.eval(eval_id, v));
    } else {
      return asin(this->val_);
    }
  }

  constexpr space eval(const std::vector<space> &values) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return asin(this->val_.eval(values));
    } else {
      return asin(this->val_);
    }
  }

  constexpr auto subs(const id_t eval_id, const space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return Asin(this->val_.subs(eval_id, v));
    } else {
      return asin(this->val_);
    }
  }

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

  constexpr space eval(const id_t eval_id, space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return acos(this->val_.eval(eval_id, v));
    } else {
      return acos(this->val_);
    }
  }

  constexpr space eval(const std::vector<space> &values) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return acos(this->val_.eval(values));
    } else {
      return acos(this->val_);
    }
  }

  constexpr auto subs(const id_t eval_id, const space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return Acos(this->val_.subs(eval_id, v));
    } else {
      return acos(this->val_);
    }
  }

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

  constexpr space eval(const id_t eval_id, space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return atan(this->val_.eval(eval_id, v));
    } else {
      return atan(this->val_);
    }
  }

  constexpr space eval(const std::vector<space> &values) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return atan(this->val_.eval(values));
    } else {
      return atan(this->val_);
    }
  }

  constexpr auto subs(const id_t eval_id, const space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return Atan(this->val_.subs(eval_id, v));
    } else {
      return atan(this->val_);
    }
  }

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

  constexpr space eval(const id_t eval_id, space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return pow(this->val_.eval(eval_id, v));
    } else {
      return pow(this->val_);
    }
  }

  constexpr space eval(const std::vector<space> &values) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return pow(this->val_.eval(values));
    } else {
      return pow(this->val_);
    }
  }

  constexpr auto subs(const id_t eval_id, const space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return Pow(this->val_.subs(eval_id, v));
    } else {
      return pow(this->val_);
    }
  }

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
