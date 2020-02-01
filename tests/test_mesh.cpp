
#include "autodiff_tao.hpp"

#include <cmath>

#include <gtest/gtest.h>

using point = std::array<double, 2>;
using vect = point;
using tri_indices = std::array<int, 3>;

point point_min(const vect &v1, const vect &v2) {
  // 1 / (3 v1 \cross v2) (v1 \cross v2 + [1, -1] v2)
  // 1 / (3 v1 \cross v2) (v1 \cross v2 - [1, -1] v2)
  const double v1xv2 = v1[0] * v2[1] - v1[1] * v2[0];
  const double mpv2 = v2[0] - v2[1];
  return std::array<double, 2>{1.0 / 3.0 + mpv2 / (3.0 * v1xv2),
                               1.0 / 3.0 - mpv2 / (3.0 * v1xv2)};
}

double theta_min(const vect &v1, const vect &v2) {
  const double theta_min_result =
      -2 *
      atan((2 * v1[0] - v1[1] - v2[0] + 2 * v2[1] -
            sqrt(5 * pow(v1[0], 2) - 8 * v1[0] * v2[0] + 6 * v1[0] * v2[1] +
                 32 * v1[0] + 5 * pow(v1[1], 2) - 6 * v1[1] * v2[0] -
                 8 * v1[1] * v2[1] - 16 * v1[1] + 5 * pow(v2[0], 2) -
                 16 * v2[0] + 5 * pow(v2[1], 2) + 32 * v2[1] + 64) +
            8) /
           (v1[0] + 2 * v1[1] - 2 * v2[0] - v2[1]));
  return theta_min_result;
}

TEST(square_test, mesh) {
  constexpr std::array<point, 5> mesh{{// First the four boundary points
                                       {-0.5, -0.5},
                                       {-0.5, 0.5},
                                       {0.5, 0.5},
                                       {0.5, -0.5},
                                       // Then the final interior point
                                       {0.25, -0.125}}};
  constexpr std::array<tri_indices, 4> pt_indices{
      {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}}};
  vect delta{0.0, 0.0};
  for (const auto &idx : pt_indices) {
    const vect v1{mesh[idx[1]][0] - mesh[idx[0]][0],
                  mesh[idx[1]][1] - mesh[idx[0]][1]};
    const vect v2{mesh[idx[2]][0] - mesh[idx[0]][0],
                  mesh[idx[2]][1] - mesh[idx[0]][1]};
    const point p0 = point_min(v1, v2);
    const double theta = theta_min(v1, v2);
    const vect v1_rot = {v1[0] * cos(theta) + v1[1] * sin(theta),
                         -v1[0] * sin(theta) + v1[1] * cos(theta)};
    const vect v2_rot = {v2[0] * cos(theta) + v2[1] * sin(theta),
                         -v2[0] * sin(theta) + v2[1] * cos(theta)};
  }
}
