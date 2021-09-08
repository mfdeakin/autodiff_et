
#ifndef AUTODIFF_HPP
#define AUTODIFF_HPP

#include <memory>

#include "autodiff_expr_wrapper.hpp"

namespace auto_diff {

using id_t = internal::id_t;

template <typename domain_t> using variable = internal::variable<domain_t>;

template <typename expr_t> using expr_domain = internal::expr_domain<expr_t>;

// expr_wrapper provides the behaviors of evaluating the function and taking the
// derivative of it wrt a specified variable
template <typename domain_t>
using expr_wrapper = internal::expr_wrapper_base<domain_t>;

template <typename expr_t,
          typename enabler = std::enable_if_t<
              std::is_base_of_v<internal::expr_type_internal, expr_t>, void>>
std::unique_ptr<expr_wrapper<expr_domain<expr_t>>> wrap_expr(const expr_t &e) {
  return std::make_unique<internal::expr_wrapper<expr_t>>(e);
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
