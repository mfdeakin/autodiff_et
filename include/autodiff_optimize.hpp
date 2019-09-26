
#ifndef AUTODIFF_OPTIMIZE_HPP
#define AUTODIFF_OPTIMIZE_HPP

#include "autodiff.hpp"

namespace auto_diff {

template <typename expr_t>
std::enable_if_t<
    std::is_base_of_v<expr, expr_t>,
    std::map<auto_diff::id_t, decltype(std::declval<expr_t>().deriv(0))>>
gradient(const expr_t &e) {
  std::set<auto_diff::id_t> var_ids;
  e.expr_vars(var_ids);
  std::map<auto_diff::id_t, decltype(std::declval<expr_t>().deriv(0))> partials;
  for (auto var : var_ids) {
    partials.insert(std::pair{var, e.deriv(var)});
  }
  return partials;
}

// Hessian implementation
// Note that the first derivative may go to a constant, so the second derivative
// will necessarily be 0 in that case. That would result in a lot of wasted
// work, so we don't implement that case and error out instead
template <typename expr_t>
std::enable_if_t<
    std::is_base_of_v<expr, expr_t> &&
        std::is_base_of_v<expr, decltype(std::declval<expr_t>().deriv(0))>,
    std::map<std::pair<auto_diff::id_t, auto_diff::id_t>,
             decltype(std::declval<expr_t>().deriv(0).deriv(0))>>
hessian(const expr_t &e) {
  std::set<auto_diff::id_t> var_ids;
  e.expr_vars(var_ids);
  std::map<std::pair<auto_diff::id_t, auto_diff::id_t>,
           decltype(std::declval<expr_t>().deriv(0).deriv(0))>
      partials;
  for (auto var_1 : var_ids) {
    for (auto var_2 : var_ids) {
      partials.insert(
          std::pair{std::pair{var_1, var_2}, e.deriv(var_1).deriv(var_2)});
    }
  }
  return partials;
}

} // namespace auto_diff

#endif // AUTODIFF_OPTIMIZE_HPP
