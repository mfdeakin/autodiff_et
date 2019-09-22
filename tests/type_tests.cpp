
#include "autodiff.hpp"
#include "autodiff_transcendental.hpp"

using namespace auto_diff;

// Check that our utility for determining the domain is correct is working
static_assert(std::is_same_v<binary_expr_domain<double, double>::space, double>,
              "expr_domain not working");
static_assert(
    std::is_same_v<binary_expr_domain<variable<double>, double>::space, double>,
    "expr_domain not working");
static_assert(
    std::is_same_v<binary_expr_domain<double, variable<double>>::space, double>,
    "expr_domain not working");
static_assert(
    std::is_same_v<
        binary_expr_domain<variable<double>, variable<double>>::space, double>,
    "expr_domain not working");

// Checks on the basic expressions
static_assert(
    std::is_same_v<decltype(-negation<variable<double>>(variable<double>(0))),
                   variable<double>>,
    "Negation isn't cancelling");

// Check that the types of the derivatives are correct and efficient
// variables derivatives
static_assert(
    std::is_same_v<decltype(variable<double>(0).deriv()), variable<double>>);

// addition derivatives
static_assert(std::is_same_v<decltype(addition<double, variable<double>>(
                                          0.0, variable<double>(0.0))
                                          .deriv()),
                             variable<double>>,
              "addition derivative type wrong");
static_assert(std::is_same_v<decltype(addition<variable<double>, double>(
                                          variable<double>(0), 0.0)
                                          .deriv()),
                             variable<double>>,
              "addition derivative type wrong");
static_assert(
    std::is_same_v<decltype(addition<variable<double>, variable<double>>(
                                variable<double>(0), variable<double>(0))
                                .deriv()),
                   addition<variable<double>, variable<double>>>,
    "addition derivative type wrong");

// subtraction derivatives
static_assert(std::is_same_v<decltype(subtraction<double, variable<double>>(
                                          0.0, variable<double>(0.0))
                                          .deriv()),
                             negation<variable<double>>>,
              "subtraction derivative type wrong");
static_assert(std::is_same_v<decltype(subtraction<variable<double>, double>(
                                          variable<double>(0), 0.0)
                                          .deriv()),
                             variable<double>>,
              "subtraction derivative type wrong");
static_assert(
    std::is_same_v<decltype(subtraction<variable<double>, variable<double>>(
                                variable<double>(0), variable<double>(0))
                                .deriv()),
                   subtraction<variable<double>, variable<double>>>,
    "subtraction derivative type wrong");

// multiplication derivatives
static_assert(std::is_same_v<decltype(multiplication<double, variable<double>>(
                                          0.0, variable<double>(0))
                                          .deriv()),
                             multiplication<double, variable<double>>>,
              "multiplication derivative type wrong");
static_assert(std::is_same_v<decltype(multiplication<variable<double>, double>(
                                          variable<double>(0), 0.0)
                                          .deriv()),
                             multiplication<variable<double>, double>>,
              "multiplication derivative type wrong");
static_assert(std::is_same_v<
                  decltype(multiplication<variable<double>, variable<double>>(
                               variable<double>(0), variable<double>(0))
                               .deriv()),
                  addition<multiplication<variable<double>, variable<double>>,
                           multiplication<variable<double>, variable<double>>>>,
              "multiplication derivative type wrong");

// division derivatives
static_assert(
    std::is_same_v<
        decltype(division<double, variable<double>>(0.0, variable<double>(0))
                     .deriv()),
        multiplication<division<double, multiplication<variable<double>,
                                                       variable<double>>>,
                       variable<double>>>,
    "division derivative type wrong");
static_assert(std::is_same_v<decltype(division<variable<double>, double>(
                                          variable<double>(0), 0.0)
                                          .deriv()),
                             division<variable<double>, double>>,
              "division derivative type wrong");
static_assert(
    std::is_same_v<
        decltype(division<variable<double>, variable<double>>(
                     variable<double>(0), variable<double>(0))
                     .deriv()),
        subtraction<division<variable<double>, variable<double>>,
                    multiplication<division<variable<double>,
                                            multiplication<variable<double>,
                                                           variable<double>>>,
                                   variable<double>>>>,
    "division derivative type wrong");

static_assert(std::is_same_v<decltype(addition<double, variable<double>>(
                                          0.0, variable<double>(0.0))
                                          .deriv()),
                             variable<double>>,
              "addition derivative type wrong");
