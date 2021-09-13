
#ifndef AUTODIFF_HPP
#define AUTODIFF_HPP

#include <memory>

#include "autodiff_expr.hpp"
#include "autodiff_expr_wrapper.hpp"

namespace auto_diff {

using id_t = internal::id_t;
constexpr id_t unit_id = internal::unit_id;

using default_domain = double;

template <typename domain_t = default_domain>
using variable = internal::variable<domain_t>;
static constexpr variable<default_domain> unit_var(unit_id);

template <typename expr_t> using expr_domain = internal::expr_domain<expr_t>;

template <typename expr_t> using is_expr = internal::is_expr<expr_t>;
template <typename expr_t>
constexpr bool is_expr_v = internal::is_expr_v<expr_t>;

template <typename expr_t> using expr_deriv = internal::expr_deriv<expr_t>;
template <typename expr_t> using expr_deriv_t = internal::expr_deriv_t<expr_t>;

template <typename expr_t>
using cond_expr_deriv = internal::cond_expr_deriv<expr_t>;
template <typename expr_t>
using cond_expr_deriv_t = internal::cond_expr_deriv_t<expr_t>;

// expr_wrapper_impl provides the behaviors of evaluating the function and
// taking the derivative of it wrt a specified variable
// Providing this container for the unique_ptr of the wrapped expression
// enables constructing larger expressions out of statically defined expressions
// Note that using it to construct larger expressions will incur heap allocation
// costs every time the copy constructor needs to be called; constructing
// expressions this way will be expensive
template <typename domain_t = default_domain>
class expr_wrapper : public internal::expr_type_internal {
public:
  using space = domain_t;

  expr_wrapper() = delete;
  explicit expr_wrapper(std::unique_ptr<internal::expr_wrapper_base<space>> &&e)
      : expr(std::move(e)) {}
  expr_wrapper(const expr_wrapper &e) : expr(e.expr->clone()) {}
  expr_wrapper(expr_wrapper &&e) : expr(std::move(e.expr)) {}
  // No expectations to inherit from this class currently exist
  ~expr_wrapper() = default;

  expr_wrapper clone() const { return expr_wrapper(expr->clone()); }

  space eval(const std::vector<space> &vals) const { return expr->eval(vals); }
  space eval(const id_t var, const space &val) const {
    return expr->eval(var, val);
  }
  space eval(const variable<space> var, const space &val) const {
    return expr->eval(var, val);
  }
  expr_wrapper deriv(const id_t var) const {
    return expr_wrapper(expr->deriv(var));
  }
  expr_wrapper deriv(const variable<space> var) const {
    return expr_wrapper(expr->deriv(var));
  }

private:
  std::unique_ptr<internal::expr_wrapper_base<space>> expr;
};

template <typename expr_t,
          typename enabler = std::enable_if_t<
              std::is_base_of_v<internal::expr_type_internal, expr_t>, void>>
expr_wrapper<expr_domain<expr_t>> wrap_expr(const expr_t &e) {
  return expr_wrapper<expr_domain<expr_t>>(
      std::make_unique<internal::expr_wrapper_impl<expr_t>>(e));
}

template <typename expr_t>
auto eval(expr_t &&e, const std::vector<expr_domain<expr_t>> &vals) {
  using space = expr_domain<expr_t>;
  if constexpr (std::is_base_of_v<internal::expr_type_internal, expr_t> ||
                std::is_base_of_v<internal::expr_wrapper_base<space>, expr_t>) {
    return e.eval(vals);
  } else {
    return e;
  }
}

template <typename expr_t>
auto eval(expr_t &&e, const id_t var, const expr_domain<expr_t> &val) {
  using space = expr_domain<expr_t>;
  if constexpr (std::is_base_of_v<internal::expr_type_internal, expr_t> ||
                std::is_base_of_v<internal::expr_wrapper_base<space>, expr_t>) {
    return e.eval(var, val);
  } else {
    return e;
  }
}

template <typename expr_t>
auto eval(expr_t &&e, const variable<expr_domain<expr_t>> var,
          const expr_domain<expr_t> &val) {
  return eval(std::forward<expr_t>(e), var.id(), val);
}

template <typename expr_t> auto deriv(expr_t &&e, const id_t var) {
  using space = expr_domain<expr_t>;
  if constexpr (std::is_base_of_v<internal::expr_type_internal, expr_t> ||
                std::is_base_of_v<internal::expr_wrapper_base<space>, expr_t>) {
    return e.deriv(var);
  } else {
    return expr_domain<expr_t>(0);
  }
}

template <typename expr_t>
auto deriv(expr_t &&e, const variable<expr_domain<expr_t>> var) {
  return deriv(std::forward<expr_t>(e), var.id());
}

} // namespace auto_diff

#endif // AUTODIFF_HPP
