
#ifndef AUTODIFF_OPTIMIZE_HPP
#define AUTODIFF_OPTIMIZE_HPP

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "autodiff_internal.hpp"

namespace auto_diff {

template <typename expr_t>
std::enable_if_t<std::is_base_of_v<internal::expr_type_internal, expr_t>,
                 std::map<auto_diff::internal::id_t,
                          decltype(std::declval<expr_t>().deriv(0))>>
gradient(const expr_t &e) {
  std::set<auto_diff::internal::id_t> var_ids;
  e.expr_vars(var_ids);
  std::map<auto_diff::internal::id_t, decltype(std::declval<expr_t>().deriv(0))>
      partials;
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
    std::is_base_of_v<internal::expr_type_internal, expr_t> &&
        std::is_base_of_v<internal::expr_type_internal,
                          decltype(std::declval<expr_t>().deriv(0))>,
    std::map<std::pair<auto_diff::internal::id_t, auto_diff::internal::id_t>,
             decltype(std::declval<expr_t>().deriv(0).deriv(0))>>
hessian(const expr_t &e) {
  std::set<auto_diff::internal::id_t> var_ids;
  e.expr_vars(var_ids);
  std::map<std::pair<auto_diff::internal::id_t, auto_diff::internal::id_t>,
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

// Constrainted non-linear minimizer
// Implements the conjugate gradient method using the Polak-Ribiere method
// The line search is exact for convex quadratic functions, and satisfies the
// strong Wolfe conditions
// PETSc's conjugate gradient solver is used to initialize the line search
// Note that we assume the Clairaut's theorem holds for these expressions
template <
    typename expr_t_,
    std::enable_if_t<std::is_base_of_v<internal::expr_type_internal, expr_t_>,
                     int> = 0>
class CNLMin {
public:
  using expr_t = expr_t_;
  using space = internal::expr_domain<expr_t>;

  static constexpr space def_tolerance =
      std::numeric_limits<space>::epsilon() * 128.0;

  CNLMin(const expr_t &f)
      : f_(f),
        f_grad_(gradient(f)), minimizer_buf{std::vector<space>(f_grad_.size()),
                                            std::vector<space>(f_grad_.size())},
        search_dir(f_grad_.size()), buffer(f_grad_.size()), origin() {}

  const expr_t &f() const { return f_; }
  const auto &f_grad() const { return f_grad_; }

  const std::vector<space> &local_minimum(
      const std::vector<space> &initial,
      const double threshold = std::numeric_limits<double>::epsilon()) {
    // Implements Fletcher-Reeves' non-linear cg method
    std::vector<space> *prev_min = &minimizer_buf[0];
    std::vector<space> *cur_min = &minimizer_buf[1];
    std::copy(initial.begin(), initial.end(), prev_min->begin());
    grad_search_dir(*prev_min, search_dir);
    double prev_sd_mag = vector_mag(search_dir);
    while (prev_sd_mag >= threshold) {
      strong_wolfe_ls(*prev_min, search_dir, *cur_min);
      grad_search_dir(*cur_min, buffer);
      const double sd_mag = vector_mag(buffer);
      const double b_k = sd_mag / prev_sd_mag;
      for (int i = 0; i < search_dir.size(); ++i) {
        search_dir[i] = buffer[i] + b_k * search_dir[i];
      }
      prev_sd_mag = sd_mag;
      std::swap(prev_min, cur_min);
    }
    return *prev_min;
  }

  const std::vector<space> &local_minimum() {
    if (origin.size() != f_grad_.size()) {
      origin = std::vector<space>(f_grad_.size());
    }
    return local_minimum(origin);
  }

  void step_pos(const std::vector<space> &x0,
                const std::vector<space> &step_dir, const double step_size,
                std::vector<space> &new_pos) const {
    for (size_t i = 0; i < x0.size(); ++i) {
      new_pos[i] = x0[i] + step_dir[i] * step_size;
    }
  }

  void grad_search_dir(const std::vector<space> &pos,
                       std::vector<space> &result) const {
    assert(result.size() == pos.size());
    assert(result.size() == f_grad_.size());
    for (size_t i = 0; i < pos.size(); ++i) {
      result[i] = -f_grad_.at(i).eval(pos);
    }
  }

  double dir_deriv(const std::vector<space> &pos,
                   const std::vector<space> &direction) const {
    double val = 0.0;
    for (size_t i = 0; i < f_grad_.size(); ++i) {
      if constexpr (std::is_base_of_v<internal::expr_type_internal,
                                      typename grad_map_type::mapped_type>) {
        val += f_grad_.at(i).eval(pos) * direction.at(i);
      } else {
        // The derivative is a constant
        val += f_grad_.at(i) * direction.at(i);
      }
    }
    return val;
  }

  double
  zoom(const std::vector<space> &x0, const std::vector<space> &step_dir,
       double step_low, double step_high, const double coeff_1,
       const double coeff_2, std::vector<space> &new_pt,
       double start_eval = std::numeric_limits<space>::quiet_NaN(),
       double low_eval = std::numeric_limits<space>::quiet_NaN(),
       double start_deriv = std::numeric_limits<space>::quiet_NaN()) const {
    // Checks to ensure the inputs are sane
    assert(0.0 < coeff_1);
    assert(coeff_1 < coeff_2);
    assert(coeff_2 < 1.0);

    // For testing; using this is inefficient since the evaluation should have
    // already checked this
    if (std::isnan(start_eval)) {
      start_eval = f_.eval(x0);
    } else {
      assert(start_eval == f_.eval(x0));
    }
    if (std::isnan(start_deriv)) {
      start_deriv = dir_deriv(x0, step_dir);
    } else {
      assert(start_deriv == dir_deriv(x0, step_dir));
    }
    assert(start_deriv < 0.0);

    step_pos(x0, step_dir, step_low, new_pt);
    if (std::isnan(low_eval)) {
      low_eval = f_.eval(new_pt);
    } else {
      assert(low_eval == f_.eval(new_pt));
    }
    const space dd_low = dir_deriv(new_pt, step_dir);

    step_pos(x0, step_dir, step_high, new_pt);
    assert(low_eval < f_.eval(new_pt));

    assert(dd_low * (step_high - step_low) < 0.0);
    // In this loop, the interval (step_low, step_high) consists partly of step
    // lengths satisfying the strong Wolfe conditions
    // The minimum function value seen is at the step length step_low
    // step_high is chosen so that the directional derivative at the minimum
    // step length times the length of the range is negative
    while (std::abs(step_low - step_high) >
               std::numeric_limits<double>::epsilon() *
                   std::max(std::abs(step_low), std::abs(step_high)) &&
           step_high > 0.0) {
      // Use bisection to choose the next trial value
      const double new_step = (step_low + step_high) / 2.0;

      step_pos(x0, step_dir, new_step, new_pt);
      const double new_eval = f_.eval(new_pt);
      if (new_eval > start_eval + coeff_1 * new_step * start_deriv ||
          new_eval >= low_eval) {
        step_high = new_step;
      } else {
        const double new_grad = dir_deriv(new_pt, step_dir);
        if (std::abs(new_grad) < -coeff_2 * start_deriv) {
          return new_step;
        }
        if (new_grad * (step_high - step_low) >= 0) {
          step_high = step_low;
        }
        step_low = new_step;
      }
    }
    return step_low;
  }

  double strong_wolfe_ls(const std::vector<space> &x0,
                         const std::vector<space> &search_dir,
                         std::vector<space> &new_pt,
                         const double max_step_size = 1.0) {
    assert(max_step_size > 0.0);
    constexpr double coeff_1 = 0.25, coeff_2 = 0.75;
    const double start_val = f_.eval(x0);
    const double start_deriv = dir_deriv(x0, search_dir);
    assert(start_deriv < 0);

    constexpr int max_steps = 16;

    double prev_step = 0.0;
    double prev_val = start_val;

    double cur_step = max_step_size / 2.0;

    for (int i = 1; i < max_steps; ++i) {
      step_pos(x0, search_dir, cur_step, new_pt);
      const double cur_val = f_.eval(new_pt);
      if ((cur_val > start_val + coeff_1 * cur_step * start_deriv) ||
          (cur_val >= prev_val && i > 1)) {
        return zoom(x0, search_dir, prev_step, cur_step, coeff_1, coeff_2,
                    new_pt, start_val, prev_val, start_deriv);
      }
      const double new_deriv = dir_deriv(new_pt, search_dir);

      if (std::abs(new_deriv) <= -coeff_2 * start_deriv) {
        return cur_step;
      }
      if (new_deriv >= 0.0) {
        return zoom(x0, search_dir, cur_step, prev_step, coeff_1, coeff_2,
                    new_pt, start_val, prev_val, start_deriv);
      }
      prev_step = cur_step;
      prev_val = cur_val;
      cur_step = (cur_step + max_step_size) / 2.0;
    }
    assert(false);
    return std::numeric_limits<space>::signaling_NaN();
  }

  const std::vector<space> &backtrack_ls(const std::vector<space> &x0,
                                         double step_size) {
    assert(x0.size() == search_dir.size());
    assert(x0.size() == f_grad_.size());

    // The wolfe_scale determines how much the step size should change after
    // each failure
    // It must be between 0 and 1
    constexpr double wolfe_scale = 0.5;
    const auto initial_val = f_.eval(x0);

    double backtrack_scale = 0.0;
    for (int i = 0; i < f_grad_.size(); ++i) {
      backtrack_scale += f_grad_[i].eval(x0) * search_dir.at(i);
    }

    std::vector<space> &new_pt = buffer;
    do {
      step_pos(x0, search_dir, step_size);
      step_size *= wolfe_scale;
    } while (f_.eval(new_pt) >
             initial_val + step_size * wolfe_scale * backtrack_scale);
    return new_pt;
  }

  static double vector_mag(const std::vector<space> &v) {
    double mag = 0.0;
    for (auto v_i : v) {
      mag += v_i * v_i;
    }
    return mag;
  }

protected:
  expr_t f_;
  using grad_map_type =
      typename std::invoke_result<decltype(gradient<expr_t>), expr_t>::type;
  grad_map_type f_grad_;
  std::array<std::vector<space>, 2> minimizer_buf;
  std::vector<space> search_dir;
  std::vector<space> buffer;
  std::vector<space> origin;
};

} // namespace auto_diff

#endif // AUTODIFF_OPTIMIZE_HPP
