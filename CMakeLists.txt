cmake_minimum_required (VERSION 3.9)
project(autodiff)

set(google_benchmark_path /usr/local CACHE PATH "Path to where Google Benchmark is installed")

enable_testing()

include_directories(include)

add_library(type_tests OBJECT EXCLUDE_FROM_ALL tests/type_tests.cpp)
set_target_properties(type_tests PROPERTIES COMPILE_FLAGS "-g -std=c++17 -Wall")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(MPI)
find_package(PETSc)

if(PETSc_FOUND AND MPI_FOUND)
  add_library(cg_solver src/solver/SyPDSolver.cpp)
  target_compile_definitions(cg_solver PUBLIC -DHAS_PETSC)
  set_target_properties(cg_solver PROPERTIES COMPILE_FLAGS "-g -std=c++17 -Wall")
  target_include_directories(cg_solver PUBLIC ${PETSC_INCLUDES} ${MPI_C_INCLUDE_DIRS})

  add_executable(petsc_tests tests/petsc_tests.cpp)
  target_compile_definitions(petsc_tests PUBLIC -DHAS_PETSC)
  set_target_properties(petsc_tests PROPERTIES COMPILE_FLAGS "-g -std=c++17 -Wall")
  target_include_directories(petsc_tests PUBLIC ${PETSC_INCLUDES} ${MPI_C_INCLUDE_DIRS})
  target_link_libraries(petsc_tests PUBLIC
    petsc cg_solver ${PETSC_LIBRARIES} gtest)
  add_test(petsc_test petsc_tests)
endif(PETSc_FOUND AND MPI_FOUND)

add_executable(unit_tests tests/test.cpp tests/test_optimize.cpp tests/test_mesh.cpp)
set_target_properties(unit_tests PROPERTIES COMPILE_FLAGS "-g -std=c++17 -Wall")
target_link_libraries(unit_tests gtest pthread)
add_test(all unit_tests)
