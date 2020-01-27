
#ifndef AUTODIFF_TAO_HPP
#define AUTODIFF_TAO_HPP

#ifdef HAS_PETSC

#include <functional>
#include <memory>
#include <set>
#include <type_traits>

#include <petsc.h>
#include <petsctao.h>

namespace auto_diff {
namespace optimize_tao {

class PetscException {
public:
  PetscException(const char *err_msg_, const PetscErrorCode err_code_)
      : err_msg(err_msg_), err_code(err_code_) {}

  const char *err_msg;
  PetscErrorCode err_code;
};

template <typename param, typename... constructor_args>
using petsc_cons = PetscErrorCode (*)(constructor_args..., param *);

template <typename param> using petsc_destr = PetscErrorCode (*)(param *);

template <typename param, typename... constructor_args>
auto petsc_make_unique(petsc_cons<param, constructor_args...> construct,
                       petsc_destr<param> destroy, constructor_args... args) {
  param p;
  const PetscErrorCode err = construct(args..., &p);
  if (err) {
    throw PetscException("Error constructing petsc object", err);
  }
  return std::unique_ptr<typename std::remove_pointer_t<param>,
                         std::function<void(param)>>(
      p, [destroy](param d) { // No exception check here, because it's unclear
                              // that anything can be done
        destroy(&d);
      });
}

template <typename expr_t> class NLOptimizer {
public:
  NLOptimizer(const expr_t f_, const size_t num_vars)
      : f(f_), tao_ctx(petsc_make_unique<Tao>(
                   static_cast<petsc_cons<Tao, MPI_Comm>>(TaoCreate),
                   static_cast<petsc_destr<Tao>>(TaoDestroy),
                   PETSC_COMM_SELF)) // ,
  // arg_min(petsc_make_unique<Vec, MPI_Comm, PetscInt>(
  //     VecCreateSeq, VecDestroy, PETSC_COMM_SELF, num_vars))
  {
    // PetscErrorCode err = TaoCreate(PETSC_COMM_SELF, &t);
    // tao_ctx = decltype(tao_ctx)(t, TaoDestructor);
    // err |= TaoSetType(tao_ctx, TAOCG);
    // if (err) {
    //   throw PetscException(
    //       "Failed to create Tao Context or set the Tao optimization method");
    // }

    std::set<id_t> vars;
    assert(f_.expr_vars(vars).size() == num_vars);

    // Vec v;
    // err = VecCreateSeq(PETSC_COMM_SELF, num_vars, &v);
    // arg_min = decltype(arg_min)(v, VecDestructor);
    // err |= VecSet(arg_min, 0.0);
    // if (err) {
    //   throw PetscException("Failed to create or set the vector");
    // }
    // err =
    //     MatCreateSeqDense(MPI_COMM_SELF, num_vars, num_vars, nullptr,
    //     &hessian);
    // if (err) {
    //   throw PetscException("Failed to create hessian");
    // }
  }

protected:
  expr_t f;

  std::unique_ptr<_p_Tao, std::function<void(Tao &)>> tao_ctx;
  std::unique_ptr<_p_Vec, std::function<void(Vec &)>> arg_min;
  std::unique_ptr<_p_Mat, std::function<void(Mat &)>> hessian;
};
} // namespace optimize_tao
} // namespace auto_diff

#endif // HAS_PETSC

#endif // AUTODIFF_TAO_HPP
