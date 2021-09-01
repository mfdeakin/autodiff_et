//
// Created by michael on 8/31/21.
//

#ifndef AUTODIFF_EXPR_HPP
#define AUTODIFF_EXPR_HPP

#include <memory>

#include "autodiff_expr.hpp"

namespace auto_diff {

namespace internal {

// This wrapper is used for virtual type erasure, without potentially causing dozens of vtable lookups
// This requires an expression constructed from the internal variable types and operators;
// the expression structure cannot be modified
// unary operator support could be added, but binary operators are not possible
template <typename space_>
class expr_wrapper_base {
public:
  using space = space_;
  using base = expr_wrapper_base<space>;

  virtual std::unique_ptr<base> copy() const = 0;
  virtual std::unique_ptr<base> deriv(const id_t &var) const = 0;
  virtual std::unique_ptr<base> deriv(const variable<space> &var) const = 0;
  virtual constexpr space eval(const std::vector<space> &vars) const = 0;

  // These are really only useful with single variable expressions; any unspecified variable is evaluated as space(0)
  virtual constexpr space eval(const id_t &eval_id, const space &v) const = 0;
  virtual constexpr space eval(const variable<space> &eval, const space &v) const = 0;
  virtual constexpr space eval(const std::pair<id_t, space> vv) const = 0;
  virtual constexpr space eval(const std::pair<variable<space>, space> vv) const = 0;
};

template <typename expr_internal>
class expr_wrapper : public expr_wrapper_base<expr_domain<expr_internal>> {
public:
  using base = expr_wrapper_base<expr_domain<expr_internal>>;
  using space = expr_domain<expr_internal>;

  expr_wrapper() = delete;
  expr_wrapper(const expr_wrapper &e) : expr_(e.expr_) {}
  explicit expr_wrapper(const expr_internal &e) : expr_(e) {}

  virtual std::unique_ptr<base> copy() const {
    return std::make_unique<expr_wrapper>(*this);
  }

  virtual std::unique_ptr<base> deriv(const id_t &var) const {
    if constexpr (std::is_base_of_v<expr_type_internal, expr_internal>) {
      auto d = expr_.deriv(var);
      auto ptr = new expr_wrapper<decltype(d)>(d);
      return std::unique_ptr<base>(ptr);
    } else {
      auto ptr = new expr_wrapper<space>(space(0));
      return std::unique_ptr<base>(ptr);
    }
  }
  virtual std::unique_ptr<base> deriv(const variable<space> &var) const {
    return deriv(var.id());
  }

  constexpr space eval(const id_t &eval_id, const space &v) const {
    if constexpr (std::is_base_of_v<expr_type_internal, expr_internal>) {
      return expr_.eval(eval_id, v);
    } else {
      return space(0);
    }
  };
  constexpr space eval(const variable<space> &eval_id, const space &v) const {
    if constexpr (std::is_base_of_v<expr_type_internal, expr_internal>) {
      return expr_.eval(eval_id, v);
    } else {
      return space(0);
    }
  }
  constexpr space eval(const std::pair<id_t, space> vv) const {
    if constexpr (std::is_base_of_v<expr_type_internal, expr_internal>) {
      return expr_.eval(vv.first, vv.second);
    } else {
      return space(0);
    }
  }
  constexpr space eval(const std::pair<variable<space>, space> vv) const {
    if constexpr (std::is_base_of_v<expr_type_internal, expr_internal>) {
      return expr_.eval(vv.first, vv.second);
    } else {
      return space(0);
    }
  }
  constexpr space eval(const std::vector<space> &values) const {
    if constexpr (std::is_base_of_v<expr_type_internal, expr_internal>) {
      return expr_.eval(values);
    } else {
      return space(0);
    }
  }
private:
  expr_internal expr_;
};

} // namespace internal

} // namespace auto_diff

#endif // AUTODIFF_EXPR_HPP
