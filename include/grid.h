#pragma once

#include <torch/extension.h>

const torch::Tensor make_3d_grid(const std::vector<float> &bb_min,
                                 const std::vector<float> &bb_max,
                                 const std::vector<int> shape);
