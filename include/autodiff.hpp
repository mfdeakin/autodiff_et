//
// Created by michael on 8/31/21.
//

#ifndef AUTODIFF_HPP
#define AUTODIFF_HPP

#include "autodiff_internal.hpp"

namespace auto_diff {

using id_t = internal::id_t;

template <typename domain_t> using variable = internal::variable<domain_t>;

} // namespace auto_diff

#endif // AUTODIFF_HPP
