#include "transform.h"

const bool normalizePoints(const torch::Tensor &points,
                           torch::Tensor &translate, float &scale) {
  const torch::Tensor min_bound = std::get<0>(torch::min(points, 0));
  const torch::Tensor max_bound = std::get<0>(torch::max(points, 0));

  translate = (min_bound + max_bound) / 2.0;

  scale = std::fmax(torch::max(max_bound - min_bound).item<float>(), 1e-6);

  return true;
}
