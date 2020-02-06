
#ifndef PETSC_HELPERS_HPP
#define PETSC_HELPERS_HPP

#ifdef HAS_PETSC

#include <petsc.h>

namespace petsc_helpers {

// This is used to put the petsc_cons type in a non-deducable context; otherwise
// this code relies on undefined behavior since petsc_cons cannot be used to
// deduce the types of constructor_args (as it's not the last set of parameters
// of the function; attempting to in this case is undefined behavior)
// Note that the standard type_identity_t should be used when switching to C++20
template <typename type_> struct type_identity { using type = type_; };
template <typename type_>
using type_identity_t = typename type_identity<type_>::type;

template <typename obj_type, typename... constructor_args>
using petsc_cons = PetscErrorCode (*)(constructor_args..., obj_type *);

template <typename obj_type> using petsc_destr = PetscErrorCode (*)(obj_type *);

// Wraps petsc types in unique pointers for RAII
template <typename obj_type, typename... constructor_args>
std::unique_ptr<std::remove_pointer_t<obj_type>, std::function<void(obj_type)>>
petsc_make_unique(
    type_identity_t<petsc_cons<obj_type, constructor_args...>> construct,
    petsc_destr<obj_type> destroy, constructor_args... args) {
  obj_type p;
  const PetscErrorCode err = construct(args..., &p);
  if (err) {
    throw std::runtime_error("Error constructing petsc object");
  }
  std::function<void(obj_type)> destroy_functor = [destroy](obj_type d) {
    PetscErrorCode err = destroy(&d);
    if (err) {
      // Presumably this never happens...
      throw std::runtime_error("Error destroying petsc object");
    }
  };
  return std::unique_ptr<std::remove_pointer_t<obj_type>,
                         std::function<void(obj_type)>>(p, destroy_functor);
}

} // namespace petsc_helpers

#endif // HAS_PETSC

#endif // PETSC_HELPERS_HPP
