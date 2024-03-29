
#ifndef AUTODIFF_EXPR_HPP
#define AUTODIFF_EXPR_HPP

#include <memory>

#include "autodiff_expr.hpp"

namespace auto_diff {

namespace internal {

// This wrapper is used for virtual type erasure, without potentially causing
// dozens of vtable lookups This requires an expression constructed from the
// internal variable types and operators; the expression structure cannot be
// modified unary operator support could be added, but binary operators are not
// possible
template <typename space_> class expr_wrapper_base : public expr_type_internal {
public:
  using space = space_;
  using base = expr_wrapper_base<space>;

  virtual ~expr_wrapper_base() = default;

  virtual std::unique_ptr<base> clone() const = 0;
  virtual std::unique_ptr<base> deriv(const id_t &var) const = 0;
  std::unique_ptr<base> deriv(const variable<space> &var) const { return deriv(var.id()); }

  virtual space eval(const std::vector<space> &vars) const = 0;

  // These are really only useful with single variable expressions; any
  // unspecified variable is evaluated as space(0)
  virtual space eval(const id_t &eval_id, const space &v) const = 0;
  space eval(const variable<space> &var, const space &val) const { return eval(var.id(), val); }
  space eval(const std::pair<id_t, space> vv) const { return eval(vv.first, vv.second); }
  space eval(const std::pair<variable<space>, space> vv) const { return eval(vv.first.id(), vv.second); }

  virtual size_t num_vars() const = 0;
  virtual std::vector<std::unique_ptr<base>> grad() const = 0;
};

template <typename expr_internal, size_t max_derivs>
class expr_wrapper_impl : public expr_wrapper_base<expr_domain<expr_internal>> {
public:
  using space = expr_domain<expr_internal>;
  using base = expr_wrapper_base<space>;

  expr_wrapper_impl() = delete;
  expr_wrapper_impl(const expr_wrapper_impl &e) : expr_(e.expr_) {}
  explicit expr_wrapper_impl(const expr_internal &e) : expr_(e) {}

  virtual std::unique_ptr<base> clone() const {
    return std::make_unique<expr_wrapper_impl>(*this);
  }

  virtual std::unique_ptr<base> deriv(const id_t &var) const {
    if constexpr (max_derivs > 0) {
      if constexpr (std::is_base_of_v<expr_type_internal, expr_internal>) {
        auto d = expr_.deriv(var);
        return std::make_unique<expr_wrapper_impl<decltype(d), max_derivs - 1>>(d);
      } else {
        // Because this is a constant, we don't have to worry about the derivatives not terminating
        return std::make_unique<expr_wrapper_impl<space, 1>>(space(0));
      }
    } else if constexpr (std::numeric_limits<space>::has_signaling_NaN) {
      return std::make_unique<expr_wrapper_impl<space, 1>>(std::numeric_limits<space>::signaling_NaN());
    } else {
      return std::make_unique<expr_wrapper_impl<space, 1>>(space(0));
    }
  }

  space eval(const id_t &eval_id, const space &v) const {
    if constexpr (std::is_base_of_v<expr_type_internal, expr_internal>) {
      return expr_.eval(eval_id, v);
    } else {
      return space(0);
    }
  };
  space eval(const std::vector<space> &values) const {
    if constexpr (std::is_base_of_v<expr_type_internal, expr_internal>) {
      return expr_.eval(values);
    } else {
      return space(0);
    }
  }

  virtual size_t num_vars() const { return 0; }
  virtual std::vector<std::unique_ptr<base>> grad() const {
    return std::vector<std::unique_ptr<base>>();
  }

private:
  expr_internal expr_;
};

} // namespace internal

} // namespace auto_diff

#endif // AUTODIFF_EXPR_HPP
