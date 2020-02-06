
#ifndef AUTODIFF_TAO_HPP
#define AUTODIFF_TAO_HPP

#ifdef HAS_PETSC

#include <functional>
#include <memory>
#include <set>
#include <type_traits>

#include <cassert>

#include "petsc_helpers.hpp"

#include <petsc.h>
#include <petsctao.h>

namespace auto_diff {
namespace optimize_tao {

template <typename expr_t> class NL_Smooth_Optimizer {
public:
  NL_Smooth_Optimizer(const expr_t f_, const size_t num_vars_)
      : f(f_), num_vars(num_vars_), x_pt(num_vars),
        tao_ctx(petsc_helpers::petsc_make_unique<Tao, MPI_Comm>(
            &TaoCreate, &TaoDestroy, PETSC_COMM_SELF)),
        arg_min(petsc_helpers::petsc_make_unique<Vec, MPI_Comm, PetscInt>(
            VecCreateSeq, VecDestroy, PETSC_COMM_SELF, num_vars_)),
        hessian(
            petsc_helpers::petsc_make_unique<Mat, MPI_Comm, int, int, double *>(
                MatCreateSeqDense, MatDestroy, PETSC_COMM_SELF, num_vars_,
                num_vars_, nullptr)) {
    PetscErrorCode err = TaoSetType(tao_ctx.get(), TAOCG);
    if (err) {
      throw std::runtime_error(
          "Failed to create Tao Context or set the Tao optimization method");
    }

    std::set<id_t> vars;
    assert(f_.expr_vars(vars).size() == num_vars);

    err = TaoSetObjectiveAndGradientRoutine(tao_ctx.get(),
                                            petscint_form_obj_grad, this);
    if (err) {
      throw std::runtime_error("Failed to set the objective/gradient routine");
    }
    err = MatSetOption(hessian.get(), MAT_SYMMETRIC, PETSC_TRUE);
    if (err) {
      throw std::runtime_error(
          "Failed to set the symmetric flag for the Hessian");
    }
    err = TaoSetHessianRoutine(tao_ctx.get(), hessian.get(), hessian.get(),
                               petscint_form_hessian, this);
    if (err) {
      throw std::runtime_error("Failed to set the Hessian routine");
    }
  }

  const std::vector<PetscScalar> &solve(const std::vector<PetscScalar> &x) {
    assert(x.size() == num_vars);
    PetscErrorCode err = VecPlaceArray(arg_min.get(), x.data());
    if (err) {
      throw std::runtime_error("Failed to set the ");
    }
    err = TaoSetInitialVector(tao_ctx.get(), arg_min.get());
    if (err) {
      throw std::runtime_error("Failed to set the initial point");
    }
    err = TaoSolve(tao_ctx.get());
    if (err) {
      throw std::runtime_error("Failed to set solve the equation");
    }
    read_pt(arg_min.get());
    return x_pt;
  }

  // fast paths are all noexcept
  void read_pt(Vec x_vec) noexcept {
    // Get the point in readable form; stores it in x_pt
    const PetscScalar *x_rd_ptr;
    PetscErrorCode err = VecGetArrayRead(x_vec, &x_rd_ptr);
    assert(!err);
    for (size_t i = 0; i < num_vars; ++i) {
      x_pt[i] = x_rd_ptr[i];
    }
    err = VecRestoreArrayRead(x_vec, &x_rd_ptr);
    assert(!err);
  }

  PetscErrorCode form_obj_grad(Vec x_vec, PetscReal *obj, Vec grad) noexcept {
    read_pt(x_vec);
    *obj = f.eval(x_pt);

    PetscErrorCode err;
    // Next evaluate the gradient
    for (size_t i = 0; i < num_vars; ++i) {
      err = VecSetValue(grad, i, f.deriv(i).eval(x_pt), INSERT_VALUES);
      assert(!err);
    }
    err = VecAssemblyBegin(grad);
    assert(!err);
    err = VecAssemblyEnd(grad);
    assert(!err);
    return 0;
  }

  PetscErrorCode form_hessian(Vec x_vec, Mat h) noexcept {
    read_pt(x_vec);

    PetscErrorCode err;
    for (size_t i = 0; i < num_vars; ++i) {
      for (size_t j = i; j < num_vars; ++j) {
        err =
            MatSetValue(h, i, j, f.deriv(i).deriv(j).eval(x_pt), INSERT_VALUES);
        assert(!err);
        if (i != j) {
          err = MatSetValue(h, i + (j - i), j, f.deriv(i).deriv(j).eval(x_pt),
                            INSERT_VALUES);
          assert(!err);
        }
      }
    }
    err = MatAssemblyBegin(h, MAT_FINAL_ASSEMBLY);
    err = MatAssemblyEnd(h, MAT_FINAL_ASSEMBLY);
    return 0;
  }

  static PetscErrorCode petscint_form_obj_grad(Tao tao, Vec x, PetscReal *obj,
                                               Vec grad,
                                               void *this_untyped) noexcept {
    NL_Smooth_Optimizer *t =
        static_cast<NL_Smooth_Optimizer<expr_t> *>(this_untyped);
    return t->form_obj_grad(x, obj, grad);
  }

  static PetscErrorCode petscint_form_hessian(Tao tao, Vec x, Mat hessian_new,
                                              Mat hessian_preconditioner,
                                              void *this_untyped) noexcept {
    NL_Smooth_Optimizer *t =
        static_cast<NL_Smooth_Optimizer<expr_t> *>(this_untyped);
    return t->form_hessian(x, hessian_new);
  }

protected:
  expr_t f;
  size_t num_vars;

  std::vector<PetscReal> x_pt;

  std::unique_ptr<_p_Tao, std::function<void(Tao)>> tao_ctx;
  std::unique_ptr<_p_Vec, std::function<void(Vec)>> arg_min;
  std::unique_ptr<_p_Mat, std::function<void(Mat)>> hessian;
};
} // namespace optimize_tao
} // namespace auto_diff

#endif // HAS_PETSC

#endif // AUTODIFF_TAO_HPP
