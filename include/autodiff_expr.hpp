
#ifndef AUTODIFF_INTERNAL_HPP
#define AUTODIFF_INTERNAL_HPP

#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <type_traits>
#include <vector>

namespace auto_diff {

namespace internal {

template <typename expr_t_> class negation;
template <typename lhs_expr_t_, typename rhs_expr_t_> class addition;
template <typename lhs_expr_t_, typename rhs_expr_t_> class subtraction;
template <typename lhs_expr_t_, typename rhs_expr_t_> class multiplication;
template <typename lhs_expr_t_, typename rhs_expr_t_> class division;

using id_t = size_t;
constexpr id_t unit_id = std::numeric_limits<id_t>::max();

// This determines the space of the variables we're working with
template <typename expr_t, typename = void> struct expr_domain_impl {
  using space = expr_t;
};

// SFINAE: The compiler will attempt to specialize to this type, to do so it
// checks whether the expr_t parameter doesn't provide a space member type, and
// if so, use this specialization, otherwise, it will use the generic
// implementation
template <typename expr_t>
struct expr_domain_impl<expr_t, std::void_t<typename expr_t::space>> {
  using space = typename expr_t::space;
};

template <typename expr_t>
using expr_domain =
    typename expr_domain_impl<std::remove_reference_t<expr_t>>::space;

// Everything that is an expression should define an alias called "space"
// indicating the domain the expression acts on
// Expressions should also publicly inherit from expr_type_internal
class expr_type_internal {};

template <typename expr_t> struct is_expr {
  static constexpr bool value = std::is_base_of_v<expr_type_internal, expr_t>;
};

template <typename expr_t> constexpr bool is_expr_v = is_expr<expr_t>::value;

template <typename expr_t,
          typename enabler = std::enable_if_t<is_expr_v<expr_t>>>
struct expr_deriv {
  using type = decltype(std::declval<expr_t &>().deriv(id_t(0)));
};

template <typename expr_t,
          typename enabler = std::enable_if_t<is_expr_v<expr_t>>>
using expr_deriv_t = typename expr_deriv<expr_t>::type;

template <typename expr_t, bool tf = is_expr_v<expr_t>> struct cond_expr_deriv {
  using type = expr_deriv_t<expr_t>;
};
template <typename expr_t> struct cond_expr_deriv<expr_t, false> {
  using type = std::tuple<>;
};
template <typename expr_t>
using cond_expr_deriv_t = typename cond_expr_deriv<expr_t>::type;

// This determines the space of the variables in the binary expression. If they
// are not the same, compilation is halted
// It may make sense to use the mathematical approach to domains, but the
// machinery required is more substantial than I'd like to deal with currently
template <typename lhs_expr_t, typename rhs_expr_t,
          typename enabler = std::enable_if_t<
              is_expr_v<lhs_expr_t> || is_expr_v<rhs_expr_t>, void>>
struct binary_expr_domain_impl {
  static_assert(
      std::is_same_v<expr_domain<lhs_expr_t>, expr_domain<rhs_expr_t>> ||
          (!is_expr_v<lhs_expr_t> &&
           std::is_convertible_v<expr_domain<lhs_expr_t>,
                                 expr_domain<rhs_expr_t>>) ||
          (!is_expr_v<rhs_expr_t> &&
           std::is_convertible_v<expr_domain<rhs_expr_t>,
                                 expr_domain<lhs_expr_t>>),
      "Invalid domains provided for the binary operands");
  using space =
      std::conditional_t<is_expr_v<lhs_expr_t>, expr_domain<lhs_expr_t>,
                         expr_domain<rhs_expr_t>>;
};

template <typename lhs_expr_t, typename rhs_expr_t>
using binary_expr_domain =
    typename binary_expr_domain_impl<lhs_expr_t, rhs_expr_t>::space;

// Determine whether the domain is indexable and the type of its index values
template <typename domain>
using index_t_impl = decltype(std::declval<domain &>()[0]);

template <typename domain, typename = void> struct is_indexable {
  static constexpr bool value = false;
  using type = void;
};

template <typename domain>
struct is_indexable<domain, std::void_t<index_t_impl<domain>>> {
  static constexpr bool value = true;
  using type = index_t_impl<domain>;
};

template <typename domain>
static constexpr bool is_indexable_v = is_indexable<domain>::value;

template <typename domain> using index_t = typename is_indexable<domain>::type;

template <typename sub_expr_t>
constexpr std::enable_if_t<is_expr_v<sub_expr_t>, negation<sub_expr_t>>
operator-(const sub_expr_t expr) {
  return negation<sub_expr_t>(expr);
}

// is_valid_binary_expr checks that at least one of the sub-expressions is an
// expression, and that the operator is defined when applied to the spaces of
// the expressions
template <typename lhs_expr_t, typename rhs_expr_t, typename bin_op>
constexpr bool is_valid_binary_expr =
    // is_invocable checks if the specified operator can be applied to the
    // specified arguments
    std::is_invocable_v<bin_op, expr_domain<lhs_expr_t>,
                        expr_domain<rhs_expr_t>> &&
    (is_expr_v<lhs_expr_t> || is_expr_v<rhs_expr_t>);

// Implement the basic binary expression operations for valid expressions
// ie, if the operation is defined for the spaces of the expressions, and if at
// least one of the components is an expression type
template <typename lhs_expr_t, typename rhs_expr_t>
constexpr std::enable_if_t<
    is_valid_binary_expr<lhs_expr_t, rhs_expr_t, std::plus<>>,
    addition<lhs_expr_t, rhs_expr_t>>
operator+(const lhs_expr_t lhs, const rhs_expr_t rhs) {
  return addition<lhs_expr_t, rhs_expr_t>(lhs, rhs);
}

template <typename lhs_expr_t, typename rhs_expr_t>
constexpr std::enable_if_t<
    is_valid_binary_expr<lhs_expr_t, rhs_expr_t, std::minus<>>,
    subtraction<lhs_expr_t, rhs_expr_t>>
operator-(const lhs_expr_t lhs, const rhs_expr_t rhs) {
  return subtraction<lhs_expr_t, rhs_expr_t>(lhs, rhs);
}

template <typename lhs_expr_t, typename rhs_expr_t>
constexpr std::enable_if_t<
    is_valid_binary_expr<lhs_expr_t, rhs_expr_t, std::multiplies<>>,
    multiplication<lhs_expr_t, rhs_expr_t>>
operator*(const lhs_expr_t lhs, const rhs_expr_t rhs) {
  return multiplication<lhs_expr_t, rhs_expr_t>(lhs, rhs);
}

template <typename lhs_expr_t, typename rhs_expr_t>
constexpr std::enable_if_t<
    is_valid_binary_expr<lhs_expr_t, rhs_expr_t, std::divides<>>,
    division<lhs_expr_t, rhs_expr_t>>
operator/(const lhs_expr_t lhs, const rhs_expr_t rhs) {
  return division<lhs_expr_t, rhs_expr_t>(lhs, rhs);
}

// The primary component of an expression
template <typename space_> class variable : public expr_type_internal {
public:
  using space = space_;

  explicit constexpr variable(const id_t id) : id_(id) {}

  // Computes the partial derivative with respect to the specified variable
  constexpr space deriv(const id_t deriv_id) const {
    if (id() == deriv_id) {
      return space(1);
    } else {
      return space(0);
    }
  }

  constexpr space deriv(const variable<space> deriv_var) const {
    return deriv(deriv_var.id());
  }

  // Evaluate the variable with the list of pairs specifying the variable and
  // the value to replace that variable with
  // Variables that are not specified are assigned the additive identity for
  // their space
  template <typename... pairs>
  constexpr space eval(const std::pair<variable<space>, space> head,
                       pairs... tail) const {
    return eval({head.first.id(), head.second});
  }

  constexpr space eval(const std::pair<variable<space>, space> head) const {
    return eval({head.first.id(), head.second});
  }

  template <typename... pairs>
  constexpr space eval(const std::pair<id_t, space> head, pairs... tail) const {
    const auto [eval_id, v] = head;
    if (eval_id == id()) {
      return v;
    } else if (id() == unit_id) {
      return space(1);
    } else {
      return eval(tail...);
    }
  }

  constexpr space eval(const std::pair<id_t, space> head) const {
    const auto [eval_id, v] = head;
    if (eval_id == id()) {
      return v;
    } else if (id() == unit_id) {
      return space(1);
    } else {
      return space(0);
    }
  }

  // A version so you don't have to pass in pairs
  template <typename... pairs>
  constexpr space eval(const id_t eval_id, space v, pairs... tail) const {
    if (eval_id == id()) {
      return v;
    } else if (id() == unit_id) {
      return space(1);
    } else {
      return eval(tail...);
    }
  }

  template <typename... pairs>
  constexpr space eval(const variable<space> eval_var, space v,
                       pairs... tail) const {
    return eval(eval_var.id(), v, tail...);
  }

  constexpr space eval(const id_t eval_id, space v) const {
    if (eval_id == id()) {
      return v;
    } else if (id() == unit_id) {
      return space(1);
    } else {
      return space(0);
    }
  }

  constexpr space eval(const variable<space> eval_var, space v) const {
    return eval(eval_var.id(), v);
  }

  // A version which accepts vectors, where the id is the entry in the vector
  // If the vector is smaller than the id, the variables are assigned the
  // additive identity for their space
  constexpr space eval(std::vector<space> values) const {
    if (id() < values.size()) {
      return values[id()];
    } else if (id() == unit_id) {
      return space(1);
    } else {
      return space(0);
    }
  }

  // Essentially the same as the previous for an indexable space
  // This is necessary to vectorize operations for indexable spaces
  template <typename Space = space, typename... pairs>
  constexpr std::enable_if_t<
      is_indexable_v<Space> && std::is_same_v<index_t<Space>, space>, space>
  eval_idx(const id_t &idx, const std::pair<id_t, Space> head,
           pairs... tail) const {
    // constexpr space eval(const std::pair<id_t, space> head, pairs... tail)
    // const {
    const auto [eval_id, v] = head;
    if (eval_id == id()) {
      return v[idx];
    } else if (id() == unit_id) {
      return space(1);
    } else {
      return eval_idx(idx, tail...);
    }
  }

  template <typename Space = space>
  constexpr std::enable_if_t<
      is_indexable_v<Space> && std::is_same_v<index_t<Space>, space>, space>
  eval_idx(const id_t &idx, const std::pair<id_t, space> head) const {
    const auto [eval_id, v] = head;
    if (eval_id == id()) {
      return v[idx];
    } else if (id() == unit_id) {
      return space(1);
    } else {
      return space(0);
    }
  }

  // A version so you don't have to pass in pairs
  template <typename Space = space, typename... pairs>
  constexpr std::enable_if_t<
      is_indexable_v<Space> && std::is_same_v<index_t<Space>, space>, space>
  eval_idx(const id_t &idx, const id_t eval_id, space v, pairs... tail) const {
    if (eval_id == id()) {
      return v[idx];
    } else if (id() == unit_id) {
      return space(1);
    } else {
      return eval_idx(idx, tail...);
    }
  }

  template <typename Space = space>
  constexpr std::enable_if_t<
      is_indexable_v<Space> && std::is_same_v<index_t<Space>, space>, space>
  eval_idx(const id_t &idx, const id_t eval_id, space v) const {
    if (eval_id == id()) {
      return v[idx];
    } else if (id() == unit_id) {
      return space(1);
    } else {
      return space(0);
    }
  }

  // A version which accepts vectors, where the id is the entry in the vector
  // If the vector is smaller than the id, the variables are assigned the
  // additive identity for their space
  template <typename Space = space>
  constexpr std::enable_if_t<
      is_indexable_v<Space> && std::is_same_v<index_t<Space>, space>, space>
  eval_idx(const id_t &idx, std::vector<space> values) const {
    if (id() <= values.size()) {
      return values[id() + 1][idx];
    } else if (id() == unit_id) {
      return space(1);
    } else {
      return space(0);
    }
  }

  // subs replaces variables in an expression with the specified value.
  // In the future, sub-expressions should also be substitutable, though they
  // may have a more complicated implementation.
  // Because the variable id is determined at runtime, we use a trick to
  // substitute it in without destroying other variables. Instead of directly
  // substituting the sub-expression in, the sub-expression is multiplied by a
  // variable. The variable may be the original variable, or it may be a special
  // variable which always evaluates to the multiplicative identity
  template <typename... pairs>
  constexpr multiplication<variable<space>, space>
  subs(const std::pair<id_t, space> head, pairs... tail) const {
    const auto [eval_id, v] = head;
    if (eval_id == id()) {
      return variable(unit_id) * v;
    } else {
      return subs(tail...);
    }
  }

  constexpr multiplication<variable<space>, space>
  subs(const std::pair<id_t, space> head) const {
    const auto [eval_id, v] = head;
    if (eval_id == id()) {
      return variable(unit_id) * v;
    } else {
      return *this * space(1);
    }
  }

  template <typename... pairs>
  constexpr multiplication<variable<space>, space>
  subs(const id_t eval_id, const space v, pairs... tail) const {
    if (eval_id == id()) {
      return variable(unit_id) * v;
    } else {
      return subs(tail...);
    }
  }

  constexpr multiplication<variable<space>, space> subs(const id_t eval_id,
                                                        const space v) const {
    if (eval_id == id()) {
      return variable(unit_id) * v;
    } else {
      return *this * space(1);
    }
  }

  constexpr std::set<id_t> &expr_vars(std::set<id_t> &var_ids) const {
    var_ids.insert(id());
    return var_ids;
  }

  constexpr id_t id() const { return id_; }

protected:
  id_t id_;
};

// Unary operation base implementation
template <typename expr_t_, typename uop_,
          std::enable_if_t<is_expr_v<expr_t_>, int> = 0>
class unary_op : public expr_type_internal {
public:
  using expr_t = expr_t_;
  using space = expr_domain<expr_t>;
  using uop = uop_;

  constexpr explicit unary_op(expr_t val) : val_(val) {}

  template <typename... pairs> constexpr auto eval(pairs... list) const {
    return uop()(this->val_.eval(list...));
  }

  constexpr space eval(const std::pair<id_t, space> head) const {
    return uop()(this->val_.eval(head));
  }

  constexpr space eval(const id_t eval_id, space v) const {
    return uop()(this->val_.eval(eval_id, v));
  }

  constexpr space eval(const std::vector<space> &values) const {
    return uop()(this->val_.eval(values));
  }

  template <typename... pairs> constexpr auto subs(pairs... list) const {
    return uop()(this->val_.subs(list...));
  }

  constexpr space subs(const std::pair<id_t, space> head) const {
    return uop()(this->val_.subs(head));
  }

  constexpr auto subs(const id_t eval_id, const space v) const {
    return uop()(this->val_.subs(eval_id, v));
  }

  constexpr std::set<id_t> &expr_vars(std::set<id_t> &var_ids) const {
    this->val_.expr_vars(var_ids);
    return var_ids;
  }

protected:
  expr_t val_;
};

template <typename expr_t_>
class negation : public unary_op<expr_t_, std::negate<expr_domain<expr_t_>>> {
public:
  using uop = unary_op<expr_t_, std::negate<expr_domain<expr_t_>>>;
  using expr_t = typename uop::expr_t;
  using space = typename uop::space;

  constexpr explicit negation(const expr_t val) : uop(val) {}

  constexpr auto deriv(const id_t deriv_id) const {
    return -this->val_.deriv(deriv_id);
  }
  constexpr space deriv(const variable<space> deriv_var) const {
    return deriv(deriv_var.id());
  }

  constexpr expr_t operator-() const { return this->val_; }
};

// Binary Operation implementation
// Specify the expression type on the left of the operator, the expression type
// on the right, and then a functor which evaluates the operation when provided
// with elements from the spaces of the left and right expressions
// Note that the binary operator will only exist if the binary operator can be
// applied to the spaces on the left and right
// The binary operation implements methods for evaluating with specified values,
// and for substituting in values without destroying the rest of the expression
// The derivative operation is left for the child class to implement, as it'll
// likely be able to implement it more efficiently
template <typename lhs_expr_t_, typename rhs_expr_t_, typename bop_,
          std::enable_if_t<is_valid_binary_expr<lhs_expr_t_, rhs_expr_t_, bop_>,
                           int> = 0>
class binary_op : public expr_type_internal {
public:
  using lhs_expr_t = lhs_expr_t_;
  using rhs_expr_t = rhs_expr_t_;
  using bop = bop_;

  using space = binary_expr_domain<lhs_expr_t_, rhs_expr_t_>;

  constexpr binary_op(const lhs_expr_t lhs_, const rhs_expr_t rhs_)
      : lhs(lhs_), rhs(rhs_) {}

  template <typename... pairs> constexpr auto eval(pairs... list) const {
    // Using if constexpr from C++17 lets us ignore the fact that lhs or rhs may
    // not provide an eval method (or any methods at all), so long as the if
    // constexpr blocks entrance to that code path
    if constexpr (is_expr_v<lhs_expr_t>) {
      if constexpr (is_expr_v<rhs_expr_t>) {
        return bop_()(this->lhs.eval(list...), this->rhs.eval(list...));
      } else {
        return bop_()(this->lhs.eval(list...), this->rhs);
      }
    } else {
      // Note that since this is a binary expression, at least one of the left
      // or right sub-expressions must actually be expr_type_internal and will
      // provide an eval method Since this branch only occurs if the left
      // sub-expression is not an expr_type_internal, the right one must be
      return bop_()(this->lhs, this->rhs.eval(list...));
    }
  }

  constexpr space eval(const std::pair<id_t, space> head) const {
    if constexpr (is_expr_v<lhs_expr_t>) {
      if constexpr (is_expr_v<rhs_expr_t>) {
        return bop_()(this->lhs.eval(head), this->rhs.eval(head));
      } else {
        return bop_()(this->lhs.eval(head), this->rhs);
      }
    } else {
      return bop_()(this->lhs, this->rhs.eval(head));
    }
  }

  constexpr space eval(const id_t eval_id, space v) const {
    if constexpr (is_expr_v<lhs_expr_t>) {
      if constexpr (is_expr_v<rhs_expr_t>) {
        return bop_()(this->lhs.eval(eval_id, v), this->rhs.eval(eval_id, v));
      } else {
        return bop_()(this->lhs.eval(eval_id, v), this->rhs);
      }
    } else {
      return bop_()(this->lhs, this->rhs.eval(eval_id, v));
    }
  }

  constexpr space eval(const std::vector<space> &values) const {
    if constexpr (is_expr_v<lhs_expr_t>) {
      if constexpr (is_expr_v<rhs_expr_t>) {
        return bop_()(this->lhs.eval(values), this->rhs.eval(values));
      } else {
        return bop_()(this->lhs.eval(values), this->rhs);
      }
    } else {
      return bop_()(this->lhs, this->rhs.eval(values));
    }
  }

  template <typename... pairs> constexpr auto subs(pairs... list) const {
    if constexpr (is_expr_v<lhs_expr_t>) {
      if constexpr (is_expr_v<rhs_expr_t>) {
        return bop_()(this->lhs.subs(list...), this->rhs.eval(list...));
      } else {
        return bop_()(this->lhs.eval(list...), this->rhs);
      }
    } else {
      return bop_()(this->lhs, this->rhs.eval(list...));
    }
  }

  constexpr auto subs(const std::pair<id_t, space> head) const {
    if constexpr (is_expr_v<lhs_expr_t>) {
      if constexpr (is_expr_v<rhs_expr_t>) {
        return bop_()(this->lhs.subs(head), this->rhs.eval(head));
      } else {
        return bop_()(this->lhs.eval(head), this->rhs);
      }
    } else {
      return bop_()(this->lhs, this->rhs.eval(head));
    }
  }

  constexpr auto subs(const id_t eval_id, const space v) const {
    if constexpr (is_expr_v<lhs_expr_t>) {
      if constexpr (is_expr_v<rhs_expr_t>) {
        return bop_()(this->lhs.subs(eval_id, v), this->rhs.eval(eval_id, v));
      } else {
        return bop_()(this->lhs.eval(eval_id, v), this->rhs);
      }
    } else {
      return bop_()(this->lhs, this->rhs.eval(eval_id, v));
    }
  }

  constexpr std::set<id_t> &expr_vars(std::set<id_t> &var_ids) const {
    if constexpr (is_expr_v<lhs_expr_t>) {
      this->lhs.expr_vars(var_ids);
    }
    if constexpr (is_expr_v<rhs_expr_t>) {
      this->rhs.expr_vars(var_ids);
    }
    return var_ids;
  }

protected:
  lhs_expr_t lhs;
  rhs_expr_t rhs;
};

// Implementations of the basic primitive operations for rings, using the stl
// functor implementations of the operations
template <typename lhs_expr_t_, typename rhs_expr_t_>
class addition : public binary_op<lhs_expr_t_, rhs_expr_t_,
                                  std::plus<expr_domain<lhs_expr_t_>>> {
public:
  using bop =
      binary_op<lhs_expr_t_, rhs_expr_t_, std::plus<expr_domain<lhs_expr_t_>>>;
  using lhs_expr_t = typename bop::lhs_expr_t;
  using rhs_expr_t = typename bop::rhs_expr_t;
  using this_t = addition<lhs_expr_t, rhs_expr_t>;
  using space = typename bop::space;

  constexpr addition(const lhs_expr_t lhs_, const rhs_expr_t rhs_)
      : bop(lhs_, rhs_) {}

  constexpr auto deriv(const id_t deriv_id) const {
    if constexpr (is_expr_v<lhs_expr_t>) {
      if constexpr (is_expr_v<rhs_expr_t>) {
        // Both the left and right expressions have non-zero derivatives
        return this->lhs.deriv(deriv_id) + this->rhs.deriv(deriv_id);
      } else {
        // Only the left expression has a non-zero derivative
        return this->lhs.deriv(deriv_id);
      }
    } else {
      // Only the right expression has a non-zero derivative
      return this->rhs.deriv(deriv_id);
    }
  }

  constexpr space deriv(const variable<space> deriv_var) const {
    return deriv(deriv_var.id());
  }
};

template <typename lhs_expr_t_, typename rhs_expr_t_>
class subtraction : public binary_op<lhs_expr_t_, rhs_expr_t_,
                                     std::minus<expr_domain<lhs_expr_t_>>> {
public:
  using bop =
      binary_op<lhs_expr_t_, rhs_expr_t_, std::minus<expr_domain<lhs_expr_t_>>>;
  using lhs_expr_t = typename bop::lhs_expr_t;
  using rhs_expr_t = typename bop::rhs_expr_t;
  using space = typename bop::space;

  constexpr subtraction(const lhs_expr_t lhs_, const rhs_expr_t rhs_)
      : bop(lhs_, rhs_) {}

  constexpr auto deriv(const id_t deriv_id) const {
    if constexpr (is_expr_v<lhs_expr_t>) {
      if constexpr (is_expr_v<rhs_expr_t>) {
        // Both the left and right expressions have non-zero derivatives
        return this->lhs.deriv(deriv_id) - this->rhs.deriv(deriv_id);
      } else {
        // Only the left expression has a non-zero derivative
        return this->lhs.deriv(deriv_id);
      }
    } else {
      // Only the right expression has a non-zero derivative
      return -this->rhs.deriv(deriv_id);
    }
  }

  constexpr space deriv(const variable<space> deriv_var) const {
    return deriv(deriv_var.id());
  }
};

template <typename lhs_expr_t_, typename rhs_expr_t_>
class multiplication
    : public binary_op<lhs_expr_t_, rhs_expr_t_,
                       std::multiplies<expr_domain<lhs_expr_t_>>> {
public:
  using bop = binary_op<lhs_expr_t_, rhs_expr_t_,
                        std::multiplies<expr_domain<lhs_expr_t_>>>;
  using lhs_expr_t = typename bop::lhs_expr_t;
  using rhs_expr_t = typename bop::rhs_expr_t;
  using space = typename bop::space;

  constexpr multiplication(const lhs_expr_t lhs_, const rhs_expr_t rhs_)
      : bop(lhs_, rhs_) {}

  constexpr auto deriv(const id_t deriv_id) const {
    if constexpr (is_expr_v<lhs_expr_t>) {
      if constexpr (is_expr_v<rhs_expr_t>) {
        // Both the left and right expressions have non-zero derivatives
        return this->lhs.deriv(deriv_id) * this->rhs +
               this->lhs * this->rhs.deriv(deriv_id);
      } else {
        // Only the left expression has a non-zero derivative
        return this->lhs.deriv(deriv_id) * this->rhs;
      }
    } else {
      // Only the right expression has a non-zero derivative
      return this->lhs * this->rhs.deriv(deriv_id);
    }
  }

  constexpr space deriv(const variable<space> deriv_var) const {
    return deriv(deriv_var.id());
  }
};

template <typename lhs_expr_t_, typename rhs_expr_t_>
class division : public binary_op<lhs_expr_t_, rhs_expr_t_,
                                  std::divides<expr_domain<lhs_expr_t_>>> {
public:
  using bop = binary_op<lhs_expr_t_, rhs_expr_t_,
                        std::divides<expr_domain<lhs_expr_t_>>>;
  using lhs_expr_t = typename bop::lhs_expr_t;
  using rhs_expr_t = typename bop::rhs_expr_t;
  using space = typename bop::space;

  constexpr division(const lhs_expr_t lhs_, const rhs_expr_t rhs_)
      : bop(lhs_, rhs_) {}

  constexpr auto deriv(const id_t deriv_id) const {
    if constexpr (is_expr_v<lhs_expr_t>) {
      if constexpr (is_expr_v<rhs_expr_t>) {
        // Both the left and right expressions have non-zero derivatives
        return this->lhs.deriv(deriv_id) / this->rhs -
               this->lhs / (this->rhs * this->rhs) * this->rhs.deriv(deriv_id);
      } else {
        // Only the left expression has a non-zero derivative
        return this->lhs.deriv(deriv_id) / this->rhs;
      }
    } else {
      // Only the right expression has a non-zero derivative
      return -this->lhs / (this->rhs * this->rhs) * this->rhs.deriv(deriv_id);
    }
  }

  constexpr space deriv(const variable<space> deriv_var) const {
    return deriv(deriv_var.id());
  }
};

} // namespace internal

} // namespace auto_diff

#endif // AUTODIFF_HPP
