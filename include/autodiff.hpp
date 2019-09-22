
#ifndef AUTODIFF_HPP
#define AUTODIFF_HPP

#include <cassert>
#include <cstddef>
#include <limits>
#include <map>
#include <type_traits>
#include <vector>

namespace auto_diff {

template <typename expr_t_> class negation;
template <typename lhs_expr_t_, typename rhs_expr_t_> class addition;
template <typename lhs_expr_t_, typename rhs_expr_t_> class subtraction;
template <typename lhs_expr_t_, typename rhs_expr_t_> class multiplication;
template <typename lhs_expr_t_, typename rhs_expr_t_> class division;

// To implement most of this at compile time while separating the expression
// from the data it works on, use the following limits. Hopefully the limit of
// taking the derivative of an expression at most 256 times isn't too
// onerous... Given that the number of possible variables (2^(64 - 8)) is more
// than the amount of memory any computer in the near (and presumably distant)
// future will have (that would need ~1e16 DRAM bits, another 30+ years of
// development if Moore's law hadn't broken), that limit shouldn't be an issue
//
// NOTE: The variable id 0 (unit_id) is reserved for a stand in of 1
using id_t = size_t;
static constexpr id_t unit_id = 0;
static constexpr id_t max_derivs = 256;
static constexpr id_t max_id = std::numeric_limits<id_t>::max() / max_derivs;
static constexpr id_t min_id = 1;

// This determines the domain/space of the variables we're working with
template <typename expr_t, typename = void> struct expr_domain {
  using space = expr_t;
};

template <typename expr_t>
struct expr_domain<expr_t, std::void_t<typename expr_t::space>> {
  using space = typename expr_t::space;
};

template <typename lhs_expr_t, typename rhs_expr_t> struct binary_expr_domain {
  static_assert(std::is_same_v<typename expr_domain<lhs_expr_t>::space,
                               typename expr_domain<rhs_expr_t>::space>,
                "Invalid domains provided for the binary operands");
  using space = typename expr_domain<lhs_expr_t>::space;
};

// Everything that is an expression should define an alias called "space"
// indicating the domain the expression acts on
// Expressions should also publically inherit from expr or expr_ops
class expr {};

template <typename sub_expr_t> class expr_ops : public expr {
public:
  constexpr negation<sub_expr_t> operator-() {
    return negation<sub_expr_t>(*static_cast<const sub_expr_t *>(this));
  }

  template <typename rhs_expr_t>
  constexpr addition<sub_expr_t, rhs_expr_t>
  operator+(const rhs_expr_t rhs) const {
    return addition<sub_expr_t, rhs_expr_t>(
        *static_cast<const sub_expr_t *>(this), rhs);
  }

  template <typename rhs_expr_t>
  constexpr subtraction<sub_expr_t, rhs_expr_t>
  operator-(const rhs_expr_t rhs) const {
    return subtraction<sub_expr_t, rhs_expr_t>(
        *static_cast<const sub_expr_t *>(this), rhs);
  }

  template <typename rhs_expr_t>
  constexpr multiplication<sub_expr_t, rhs_expr_t>
  operator*(const rhs_expr_t rhs) const {
    return multiplication<sub_expr_t, rhs_expr_t>(
        *static_cast<const sub_expr_t *>(this), rhs);
  }

  template <typename rhs_expr_t>
  constexpr division<sub_expr_t, rhs_expr_t>
  operator/(const rhs_expr_t rhs) const {
    return division<sub_expr_t, rhs_expr_t>(
        *static_cast<const sub_expr_t *>(this), rhs);
  }
};

template <typename rhs_expr_t>
constexpr addition<typename rhs_expr_t::space, rhs_expr_t>
operator+(const typename rhs_expr_t::space c, const rhs_expr_t e) {
  return addition<typename rhs_expr_t::space, rhs_expr_t>(c, e);
}

template <typename rhs_expr_t>
constexpr subtraction<typename rhs_expr_t::space, rhs_expr_t>
operator-(const typename rhs_expr_t::space c, const rhs_expr_t e) {
  return subtraction<typename rhs_expr_t::space, rhs_expr_t>(c, e);
}

template <typename rhs_expr_t>
constexpr multiplication<typename rhs_expr_t::space, rhs_expr_t>
operator*(const typename rhs_expr_t::space c, const rhs_expr_t e) {
  return multiplication<typename rhs_expr_t::space, rhs_expr_t>(c, e);
}

template <typename rhs_expr_t>
constexpr division<typename rhs_expr_t::space, rhs_expr_t>
operator/(const typename rhs_expr_t::space c, const rhs_expr_t e) {
  return division<typename rhs_expr_t::space, rhs_expr_t>(c, e);
}

// To implement this, I'm assuming that all variables have derivatives taken wrt
// some parameterization of a curve, ie, given an expression u(x1, x2, ..., xn),
// du/dt = du/dx1 dx1/dt + du/dx2 dx2/dt + ... + du/dxn dxn/dt
// Then to compute du/dxk, specify dxk/dt = 1 and the rest as 0
// This enables representing the gradient and Hessian in relatively concise
// forms, as the gradient is represented directly as above, and the Hessian as
// d^2u/dt^2 = du/dx1 d^2x1/dt^2 +  d^2u/dx1dxk dxk/dt dx1/dt + ...
// Then to compute d^2u/dxi^2, set d^2xi/dt^2 = 1 and the rest as zero
// Then to compute d^2u/dxidxj, specify dxi/dt = dxj/dt = 1 and the rest as zero
// This leaves us with d^2u/dxi^2 + d^2u/dxj^2 + d^2u/dxidxj + d^2u/dxjdxi
// The extra first two parts can be subtracted out since they're easy to
// compute, leaving us with the components of the Hessian
// In most cases we're interested in (where Clairaut's theorem holds),
// d^2u/dxidxj = d^2u/dxjdxi, so we just divide the remainder by 2 to get
// d^2u/dxidxj
// In the cases where this isn't true, you can use the subs method, which only
// replaces the specified variable with a constant; this
template <typename space_> class variable : public expr_ops<variable<space_>> {
public:
  using space = space_;

  explicit constexpr variable(const id_t &id) : id_(id) {}

  constexpr variable deriv() const { return variable(deriv_id(1)); }

  constexpr space eval(const id_t eval_id, space v) const {
    if (eval_id == id()) {
      return v;
    } else if (unit_id == id()) {
      return space(1);
    } else {
      return space(0);
    }
  }

  constexpr space eval(std::vector<space> values) const {
    if (id() <= values.size()) {
      return values[id() + 1];
    } else if (id() == unit_id) {
      return space(1);
    } else {
      return space(0);
    }
  }

  constexpr multiplication<variable<space>, space> subs(const id_t eval_id,
                                                        const space v) const {
    if (eval_id == id()) {
      return variable(0) * v;
    } else {
      return variable(0) * space(1);
    }
  }

  constexpr bool uses_id(id_t eval_id) const { return id() == eval_id; }

  constexpr id_t id() const { return id_; }
  constexpr id_t deriv_id(unsigned int deriv_order) const {
    assert(id() + max_id * deriv_order >= id());
    return id() + max_id * deriv_order;
  }

protected:
  id_t id_;
};

// Primitive Unary operations
template <typename expr_t_> class unary_op {
public:
  using expr_t = expr_t_;
  using space = typename expr_domain<expr_t>::space;

  constexpr explicit unary_op(expr_t val) : val_(val) {}

protected:
  expr_t val_;
};

template <typename expr_t_>
class negation : public unary_op<expr_t_>, public expr_ops<negation<expr_t_>> {
public:
  using uop = unary_op<expr_t_>;
  using expr_t = typename uop::expr_t;
  using space = typename uop::space;

  constexpr explicit negation(expr_t val) : uop(val) {}

  constexpr space eval(const id_t eval_id, space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return -this->val_.eval(eval_id, v);
    } else {
      return -this->val_;
    }
  }

  constexpr space eval(const std::vector<space> &values) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return -this->val_.eval(values);
    } else {
      return -this->val_;
    }
  }

  constexpr auto subs(const id_t eval_id, const space v) const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return -this->val_.subs(eval_id, v);
    } else {
      return -this->val_;
    }
  }

  constexpr auto deriv() const {
    if constexpr (std::is_base_of_v<expr, expr_t>) {
      return -this->val_.deriv();
    } else {
      return space(0);
    }
  }
};

// Primitive Binary Operation
template <typename lhs_expr_t_, typename rhs_expr_t_> class binary_op {
public:
  using lhs_expr_t = lhs_expr_t_;
  using rhs_expr_t = rhs_expr_t_;

  using space = typename binary_expr_domain<lhs_expr_t_, rhs_expr_t_>::space;

  constexpr binary_op(lhs_expr_t lhs_, rhs_expr_t rhs_)
      : lhs(lhs_), rhs(rhs_) {}

protected:
  lhs_expr_t lhs;
  rhs_expr_t rhs;
};

template <typename lhs_expr_t_, typename rhs_expr_t_>
class addition : public binary_op<lhs_expr_t_, rhs_expr_t_>,
                 public expr_ops<addition<lhs_expr_t_, rhs_expr_t_>> {
public:
  using bop = binary_op<lhs_expr_t_, rhs_expr_t_>;
  using lhs_expr_t = typename bop::lhs_expr_t;
  using rhs_expr_t = typename bop::rhs_expr_t;
  using this_t = addition<lhs_expr_t, rhs_expr_t>;
  using space = typename bop::space;

  constexpr addition(lhs_expr_t lhs_, rhs_expr_t rhs_) : bop(lhs_, rhs_) {}

  constexpr space eval(const id_t eval_id, space v) const {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs.eval(eval_id, v) + this->rhs.eval(eval_id, v);
      } else {
        return this->lhs.eval(eval_id, v) + this->rhs;
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs + this->rhs.eval(eval_id, v);
      } else {
        return this->lhs + this->rhs;
      }
    }
  }

  constexpr space eval(const std::vector<space> &values) const {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs.eval(values) + this->rhs.eval(values);
      } else {
        return this->lhs.eval(values) + this->rhs;
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs + this->rhs.eval(values);
      } else {
        return this->lhs + this->rhs;
      }
    }
  }

  constexpr auto subs(const id_t eval_id, const space v) const {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs.subs(eval_id, v) + this->rhs.eval(eval_id, v);
      } else {
        return this->lhs.eval(eval_id, v) + this->rhs;
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs + this->rhs.eval(eval_id, v);
      } else {
        return this->lhs + this->rhs;
      }
    }
  }

  auto deriv() {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        // Both the left and right expressions have non-zero derivatives
        return this->lhs.deriv() + this->rhs.deriv();
      } else {
        // Only the left expression has a non-zero derivative
        return this->lhs.deriv();
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        // Only the right expression has a non-zero derivative
        return this->rhs.deriv();
      } else {
        // Both expressions have a zero derivatives
        return space(0);
      }
    }
  }
};

template <typename lhs_expr_t_, typename rhs_expr_t_>
class subtraction : public binary_op<lhs_expr_t_, rhs_expr_t_>,
                    public expr_ops<subtraction<lhs_expr_t_, rhs_expr_t_>> {
public:
  using bop = binary_op<lhs_expr_t_, rhs_expr_t_>;
  using lhs_expr_t = typename bop::lhs_expr_t;
  using rhs_expr_t = typename bop::rhs_expr_t;
  using space = typename bop::space;

  constexpr subtraction(lhs_expr_t lhs_, rhs_expr_t rhs_) : bop(lhs_, rhs_) {}

  constexpr space eval(const id_t eval_id, space v) const {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs.eval(eval_id, v) - this->rhs.eval(eval_id, v);
      } else {
        return this->lhs.eval(eval_id, v) - this->rhs;
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs - this->rhs.eval(eval_id, v);
      } else {
        return this->lhs - this->rhs;
      }
    }
  }

  constexpr space eval(const std::vector<space> &values) const {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs.eval(values) - this->rhs.eval(values);
      } else {
        return this->lhs.eval(values) - this->rhs;
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs - this->rhs.eval(values);
      } else {
        return this->lhs - this->rhs;
      }
    }
  }

  constexpr auto subs(const id_t eval_id, const space v) const {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs.subs(eval_id, v) - this->rhs.eval(eval_id, v);
      } else {
        return this->lhs.eval(eval_id, v) - this->rhs;
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs - this->rhs.eval(eval_id, v);
      } else {
        return this->lhs - this->rhs;
      }
    }
  }

  auto deriv() {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        // Both the left and right expressions have non-zero derivatives
        return this->lhs.deriv() - this->rhs.deriv();
      } else {
        // Only the left expression has a non-zero derivative
        return this->lhs.deriv();
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        // Only the right expression has a non-zero derivative
        return -this->rhs.deriv();
      } else {
        // Both expressions have a zero derivatives
        return space(0);
      }
    }
  }
};

template <typename lhs_expr_t_, typename rhs_expr_t_>
class multiplication
    : public binary_op<lhs_expr_t_, rhs_expr_t_>,
      public expr_ops<multiplication<lhs_expr_t_, rhs_expr_t_>> {
public:
  using bop = binary_op<lhs_expr_t_, rhs_expr_t_>;
  using lhs_expr_t = typename bop::lhs_expr_t;
  using rhs_expr_t = typename bop::rhs_expr_t;
  using space = typename bop::space;

  constexpr multiplication(lhs_expr_t lhs_, rhs_expr_t rhs_)
      : bop(lhs_, rhs_) {}

  constexpr space eval(const id_t eval_id, space v) const {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs.eval(eval_id, v) * this->rhs.eval(eval_id, v);
      } else {
        return this->lhs.eval(eval_id, v) * this->rhs;
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs * this->rhs.eval(eval_id, v);
      } else {
        return this->lhs * this->rhs;
      }
    }
  }

  constexpr space eval(const std::vector<space> &values) const {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs.eval(values) * this->rhs.eval(values);
      } else {
        return this->lhs.eval(values) * this->rhs;
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs * this->rhs.eval(values);
      } else {
        return this->lhs * this->rhs;
      }
    }
  }

  constexpr auto subs(const id_t eval_id, const space v) const {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs.subs(eval_id, v) * this->rhs.eval(eval_id, v);
      } else {
        return this->lhs.eval(eval_id, v) * this->rhs;
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs * this->rhs.eval(eval_id, v);
      } else {
        return this->lhs * this->rhs;
      }
    }
  }

  constexpr auto deriv() {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        // Both the left and right expressions have non-zero derivatives
        return this->lhs.deriv() * this->rhs + this->lhs * this->rhs.deriv();
      } else {
        // Only the left expression has a non-zero derivative
        return this->lhs.deriv() * this->rhs;
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        // Only the right expression has a non-zero derivative
        return this->lhs * this->rhs.deriv();
      } else {
        // Both expressions have a zero derivatives
        return space(0);
      }
    }
  }
};

template <typename lhs_expr_t_, typename rhs_expr_t_>
class division : public binary_op<lhs_expr_t_, rhs_expr_t_>,
                 public expr_ops<division<lhs_expr_t_, rhs_expr_t_>> {
public:
  using bop = binary_op<lhs_expr_t_, rhs_expr_t_>;
  using lhs_expr_t = typename bop::lhs_expr_t;
  using rhs_expr_t = typename bop::rhs_expr_t;
  using space = typename bop::space;

  constexpr division(lhs_expr_t lhs_, rhs_expr_t rhs_) : bop(lhs_, rhs_) {}

  constexpr space eval(const id_t eval_id, space v) const {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs.eval(eval_id, v) / this->rhs.eval(eval_id, v);
      } else {
        return this->lhs.eval(eval_id, v) / this->rhs;
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs / this->rhs.eval(eval_id, v);
      } else {
        return this->lhs / this->rhs;
      }
    }
  }

  constexpr space eval(const std::vector<space> &values) const {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs.eval(values) / this->rhs.eval(values);
      } else {
        return this->lhs.eval(values) / this->rhs;
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs / this->rhs.eval(values);
      } else {
        return this->lhs / this->rhs;
      }
    }
  }

  constexpr auto subs(const id_t eval_id, const space v) const {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs.subs(eval_id, v) / this->rhs.eval(eval_id, v);
      } else {
        return this->lhs.eval(eval_id, v) / this->rhs;
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        return this->lhs / this->rhs.eval(eval_id, v);
      } else {
        return this->lhs / this->rhs;
      }
    }
  }

  constexpr auto deriv() {
    if constexpr (std::is_base_of_v<expr, lhs_expr_t>) {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        // Both the left and right expressions have non-zero derivatives
        return this->lhs.deriv() / this->rhs -
               this->lhs / (this->rhs * this->rhs) * this->rhs.deriv();
      } else {
        // Only the left expression has a non-zero derivative
        return this->lhs.deriv() / this->rhs;
      }
    } else {
      if constexpr (std::is_base_of_v<expr, rhs_expr_t>) {
        // Only the right expression has a non-zero derivative
        return -this->lhs / (this->rhs * this->rhs) * this->rhs.deriv();
      } else {
        // Both expressions have a zero derivatives
        return space(0);
      }
    }
  }
};

} // namespace auto_diff

#endif // AUTODIFF_HPP
