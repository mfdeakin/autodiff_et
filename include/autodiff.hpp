//
// Created by michael on 8/31/21.
//

#ifndef AUTODIFF_HPP
#define AUTODIFF_HPP

#include <memory>

#include "autodiff_expr_wrapper.hpp"

namespace auto_diff {

using id_t = internal::id_t;

template <typename domain_t> using variable = internal::variable<domain_t>;

template <typename expr_t> using expr_domain = internal::expr_domain<expr_t>;

// expr_wrapper provides the behaviors of evaluating the function and taking the derivative of it wrt a specified variable
template <typename domain_t>
using expr_wrapper = internal::expr_wrapper_base<domain_t>;

template <typename expr_t,
    typename enabler = std::enable_if_t<
        std::is_base_of_v<internal::expr_type_internal, expr_t>,
        void>>
std::unique_ptr<expr_wrapper<expr_domain<expr_t>>> wrap_expr(const expr_t &e) {
  // /usr/lib/gcc/x86_64-redhat-linux/11/../../../../include/c++/11/bits/unique_ptr.h:962:34: error: call to constructor of 'auto_diff::internal::expr_wrapper<auto_diff::internal::addition<auto_diff::internal::variable<double>, double>>' is ambiguous [clang-diagnostic-error]

  return std::make_unique<internal::expr_wrapper<expr_t>>(e);
}

} // namespace auto_diff

#endif // AUTODIFF_HPP
