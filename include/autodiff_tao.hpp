
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

// This is to put the petsc_cons type in a non-deducable context; otherwise this
// code relies on undefined behavior since petsc_cons cannot be used to deduce
// the types of constructor_args (as it's not the last set of parameters of the
// function; attempting to in this case is undefined behavior)
// Note that the standard type_identity_t should be used when switching to C++20
template <typename type_> struct type_identity { using type = type_; };
template <typename type_>
using type_identity_t = typename type_identity<type_>::type;

template <typename obj_type, typename... constructor_args>
using petsc_cons = PetscErrorCode (*)(constructor_args..., obj_type *);

template <typename obj_type> using petsc_destr = PetscErrorCode (*)(obj_type *);

// Wraps petsc types in unique pointers for RAII
template <typename obj_type, typename... constructor_args>
auto petsc_make_unique(
    type_identity_t<petsc_cons<obj_type, constructor_args...>> construct,
    petsc_destr<obj_type> destroy, constructor_args... args) {
  obj_type p;
  const PetscErrorCode err = construct(args..., &p);
  if (err) {
    throw std::runtime_error("Error constructing petsc object");
  }
  auto destroy_functor = [destroy](obj_type d) {
    PetscErrorCode err = destroy(&d);
    if (err) {
      // Presumably this never happens...
      throw std::runtime_error("Error destroying petsc object");
    }
  };
  return std::unique_ptr<typename std::remove_pointer_t<obj_type>,
                         std::function<void(obj_type)>>(p, destroy_functor);
}

template <typename expr_t> class NLOptimizer {
public:
  NLOptimizer(const expr_t f_, const size_t num_vars)
      : f(f_), tao_ctx(petsc_make_unique<Tao, MPI_Comm>(&TaoCreate, &TaoDestroy,
                                                        PETSC_COMM_SELF)),
        arg_min(petsc_make_unique<Vec, MPI_Comm, PetscInt>(
            VecCreateSeq, VecDestroy, PETSC_COMM_SELF, num_vars)),
        hessian(petsc_make_unique<Mat, MPI_Comm, int, int, void *>(
            MatCreateSeqDense, MatDestroy, PETSC_COMM_SELF, num_vars, num_vars,
            nullptr)) {
    PetscErrorCode err = TaoSetType(tao_ctx.get(), TAOCG);
    if (err) {
      throw std::runtime_error(
          "Failed to create Tao Context or set the Tao optimization method");
    }

    std::set<id_t> vars;
    assert(f_.expr_vars(vars).size() == num_vars);

    err = VecSet(arg_min.get(), 0.0);
    if (err) {
      throw std::runtime_error("Failed to create or set the vector");
    }
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
