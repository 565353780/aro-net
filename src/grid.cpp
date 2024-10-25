#include "grid.h"

const torch::Tensor make_3d_grid(const std::vector<float> &bb_min,
                                 const std::vector<float> &bb_max,
                                 const std::vector<int> shape) {
  if (shape.size() != 3) {
    std::cout << "[ERROR][grid::make_3d_grid]" << std::endl;
    std::cout << "\t shape dim != 3!" << std::endl;

    return torch::Tensor();
  }

  const int size = shape[0] * shape[1] * shape[2];

  torch::Tensor pxs = torch::linspace(bb_min[0], bb_max[0], shape[0]);
  torch::Tensor pys = torch::linspace(bb_min[1], bb_max[1], shape[1]);
  torch::Tensor pzs = torch::linspace(bb_min[2], bb_max[2], shape[2]);

  pxs = pxs.view({-1, 1, 1})
            .expand({shape[0], shape[1], shape[2]})
            .contiguous()
            .view(size);

  pys = pys.view({1, -1, 1})
            .expand({shape[0], shape[1], shape[2]})
            .contiguous()
            .view(size);

  pzs = pzs.view({1, 1, -1})
            .expand({shape[0], shape[1], shape[2]})
            .contiguous()
            .view(size);

  const torch::Tensor p = torch::stack({pxs, pys, pzs}, 1);

  return p;
}
