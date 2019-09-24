
#include "autodiff.hpp"
#include "autodiff_transcendental.hpp"

using namespace auto_diff;

// Check that our utility for determining the domain is correct is working
static_assert(std::is_same_v<binary_expr_domain<double, double>, double>,
              "expr_domain not working");
static_assert(
    std::is_same_v<binary_expr_domain<variable<double>, double>, double>,
    "expr_domain not working");
static_assert(
    std::is_same_v<binary_expr_domain<double, variable<double>>, double>,
    "expr_domain not working");
static_assert(
    std::is_same_v<binary_expr_domain<variable<double>, variable<double>>,
                   double>,
    "expr_domain not working");

// Checks that is_valid_binary_expr is working on some basic types
static_assert(!is_valid_binary_expr<double, double, std::plus<>>,
              "is_valid_binary_expr let non-expressions through");
static_assert(!is_valid_binary_expr<double, double, std::minus<>>,
              "is_valid_binary_expr let non-expressions through");
static_assert(!is_valid_binary_expr<double, double, std::multiplies<>>,
              "is_valid_binary_expr let non-expressions through");
static_assert(!is_valid_binary_expr<double, double, std::divides<>>,
              "is_valid_binary_expr let non-expressions through");

static_assert(is_valid_binary_expr<variable<double>, double, std::plus<>>,
              "is_valid_binary_expr failed on a valid expression");
static_assert(is_valid_binary_expr<double, variable<double>, std::plus<>>,
              "is_valid_binary_expr failed on a valid expression");
static_assert(
    is_valid_binary_expr<variable<double>, variable<double>, std::plus<>>,
    "is_valid_binary_expr failed on a valid expression");

// Checks on the basic expressions
static_assert(
    std::is_same_v<decltype(-negation<variable<double>>(variable<double>(0))),
                   variable<double>>,
    "Negation isn't cancelling");

// Check that the types of the derivatives are correct and efficient
// variables derivatives
static_assert(std::is_same_v<decltype(variable<double>(0).deriv(0)), double>);

// addition derivatives
static_assert(std::is_same_v<decltype(addition<double, variable<double>>(
                                          0.0, variable<double>(0.0))
                                          .deriv(0)),
                             double>,
              "addition derivative type wrong");
static_assert(std::is_same_v<decltype(addition<variable<double>, double>(
                                          variable<double>(0), 0.0)
                                          .deriv(0)),
                             double>,
              "addition derivative type wrong");
static_assert(
    std::is_same_v<decltype(addition<variable<double>, variable<double>>(
                                variable<double>(0), variable<double>(0))
                                .deriv(0)),
                   double>,
    "addition derivative type wrong");

// subtraction derivatives
static_assert(std::is_same_v<decltype(subtraction<double, variable<double>>(
                                          0.0, variable<double>(0.0))
                                          .deriv(0)),
                             double>,
              "subtraction derivative type wrong");
static_assert(std::is_same_v<decltype(subtraction<variable<double>, double>(
                                          variable<double>(0), 0.0)
                                          .deriv(0)),
                             double>,
              "subtraction derivative type wrong");
static_assert(
    std::is_same_v<decltype(subtraction<variable<double>, variable<double>>(
                                variable<double>(0), variable<double>(0))
                                .deriv(0)),
                   double>,
    "subtraction derivative type wrong");

// multiplication derivatives
static_assert(std::is_same_v<decltype(multiplication<double, variable<double>>(
                                          0.0, variable<double>(0))
                                          .deriv(0)),
                             double>,
              "multiplication derivative type wrong");
static_assert(std::is_same_v<decltype(multiplication<variable<double>, double>(
                                          variable<double>(0), 0.0)
                                          .deriv(0)),
                             double>,
              "multiplication derivative type wrong");
static_assert(
    std::is_same_v<decltype(multiplication<variable<double>, variable<double>>(
                                variable<double>(0), variable<double>(0))
                                .deriv(0)),
                   addition<multiplication<double, variable<double>>,
                            multiplication<variable<double>, double>>>,
    "multiplication derivative type wrong");

// division derivatives
static_assert(
    std::is_same_v<
        decltype(division<double, variable<double>>(0.0, variable<double>(0))
                     .deriv(0)),
        multiplication<division<double, multiplication<variable<double>,
                                                       variable<double>>>,
                       double>>,
    "division derivative type wrong");
static_assert(std::is_same_v<decltype(division<variable<double>, double>(
                                          variable<double>(0), 0.0)
                                          .deriv(0)),
                             double>,
              "division derivative type wrong");
static_assert(
    std::is_same_v<
        decltype(division<variable<double>, variable<double>>(
                     variable<double>(0), variable<double>(0))
                     .deriv(0)),
        subtraction<division<double, variable<double>>,
                    multiplication<division<variable<double>,
                                            multiplication<variable<double>,
                                                           variable<double>>>,
                                   double>>>,
    "division derivative type wrong");
