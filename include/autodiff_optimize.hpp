
#ifndef AUTODIFF_OPTIMIZE_HPP
#define AUTODIFF_OPTIMIZE_HPP

#include "SyPDSolver.hpp"
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

// Constrainted non-linear minimizer
// Implements the conjugate gradient method using the Polak-Ribiere method
// The line search is exact for convex quadratic functions, and satisfies the
// strong Wolfe conditions
// PETSc's conjugate gradient solver is used to initialize the line search
// Note that we assume the Clairaut's theorem holds for these expressions
template <typename expr_t_,
          std::enable_if_t<std::is_base_of_v<expr, expr_t_>, int> = 0>
class CNLMin {
public:
  using expr_t = expr_t_;
  using space = expr_domain<expr_t>;

  static constexpr space def_tolerance =
      std::numeric_limits<space>::epsilon() * 128.0;

  CNLMin(const expr_t &f)
      : f_(f), f_grad_(gradient(f)), minimizer(f_grad_.size()),
        buffer(f_grad_.size()), origin() {}

  const expr_t &f() const { return f_; }
  const auto &f_grad() const { return f_grad_; }
  const std::vector<space> &prev_minimizer() const { return minimizer; }

  const std::vector<space> &local_minimum(const std::vector<space> &initial) {
    return minimizer;
  }

  const std::vector<space> &local_minimum() {
    if (origin.size() != f_grad_.size()) {
      origin = std::vector<space>(f_grad_.size());
    }
    return local_minimum(origin);
  }

protected:
  void step_pos(const std::vector<space> &x0,
                const std::vector<space> &step_dir, const double step_size,
                std::vector<space> &new_pos) {
    for (int i = 0; i < x0.size(); ++i) {
      new_pos[i] = x0[i] + step_dir[i] * step_size;
    }
  }

  double eval_grad(const std::vector<space> &pos) const {
    double val = 0.0;
    for (auto pderiv : f_grad_) {
      val += pderiv.eval(pos);
    }
    return val;
  }

  void zoom(std::vector<space> &x0, std::vector<space> &step_dir,
            double step_min, double step_max, const double start_val,
            const double low_val, const double start_deriv,
            const double coeff_1, const double coeff_2,
            std::vector<space> &new_pt) {
    for (;;) {
      const double new_step = (step_min + step_max) / 2.0;
      step_pos(x0, step_dir, new_step, new_pt);
      const double new_eval = f_.eval(new_pt);
      if (new_eval > start_val + coeff_1 * new_step * start_deriv ||
          new_eval >= low_val) {
        step_pos(x0, step_dir, step_max, new_pt);
      } else {
        const double new_grad = eval_grad(new_pt);
        if (std::abs(new_grad) < -coeff_2 * start_deriv) {
          return;
        }
        if (new_grad * (step_max - step_min) <= 0) {
          step_max = step_min;
        }
        step_min = new_step;
      }
    }
  }

  const std::vector<space> &
  strong_wolfe_ls(const std::vector<space> &x0,
                  const std::vector<space> &search_dir,
                  const double max_step_size = 1.0) {
    assert(max_step_size > 0.0);
    constexpr double coeff_1 = 0.5, coeff_2 = 0.5;
    const double start_val = f_.eval(x0);
    const double start_deriv = eval_grad(x0);

    double prev_step = 0.0;
    double cur_step = max_step_size;

    std::vector<space> &new_pt = buffer;
    double prev_val = std::numeric_limits<double>::infinity();
    for (;;) {
      step_pos(x0, search_dir, cur_step, new_pt);
      const double cur_val = f_.eval(new_pt);
      if ((cur_val > x0 + cur_step * coeff_1 * start_deriv) ||
          (cur_val >= prev_val)) {
        zoom(prev_step, cur_step, start_val, prev_val, start_deriv, coeff_1,
             coeff_2, new_pt);
        return new_pt;
      }
      const double new_deriv = eval_grad(new_pt);

      if (std::abs(new_deriv) <= -coeff_2 * start_deriv) {
        return new_pt;
      }
      if (new_deriv >= 0.0) {
        zoom(cur_step, prev_step, start_val, cur_val, start_deriv, coeff_1,
             coeff_2, new_pt);
        return new_pt;
      }
      prev_val = cur_val;
    }
  }

  const std::vector<space> &backtrack_ls(const std::vector<space> &x0,
                                         const std::vector<space> &search_dir,
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

  expr_t f_;
  typename std::invoke_result<decltype(gradient<expr_t>), expr_t>::type f_grad_;
  std::vector<space> minimizer;
  std::vector<space> buffer;
  std::vector<space> origin;
};

} // namespace auto_diff

#endif // AUTODIFF_OPTIMIZE_HPP
