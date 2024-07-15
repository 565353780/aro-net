#pragma once

#include <torch/extension.h>

const bool normalizePoints(const torch::Tensor &points,
                           torch::Tensor &translate, float &scale);
