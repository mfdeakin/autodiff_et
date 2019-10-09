cmake_minimum_required (VERSION 3.9)
project(autodiff)

set(google_benchmark_path /usr/local CACHE PATH "Path to where Google Benchmark is installed")

enable_testing()

#set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
include_directories(include
  /home/michael/programming/petsc/include
  /home/michael/programming/petsc/arch-linux2-c-debug/include)

add_library(cg_solver src/solver/SyPDSolver.cpp)

add_executable(unit_tests tests/test.cpp tests/type_tests.cpp)
set_target_properties(unit_tests PROPERTIES COMPILE_FLAGS "-g -std=c++17 -Wall")
target_link_directories(unit_tests PUBLIC /home/michael/programming/petsc/arch-linux2-c-debug/lib)
target_link_libraries(unit_tests gtest pthread petsc cg_solver)
add_test(all unit_tests)
