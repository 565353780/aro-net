#include "grid.h"
#include <ATen/ops/from_blob.h>
#include <torch/types.h>
#define MC_IMPLEM_ENABLE
#include "MC.h"
#include "cnpy.h"
#include "detector.h"
#include "transform.h"
#include <filesystem>
#include <memory>
#include <torch/script.h>

using namespace torch::indexing;

Detector::Detector(const std::string &anchor_file_path,
                   const std::string &model_file_path, const bool &use_gpu) {
  if (use_gpu) {
    opts_ = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  } else {
    opts_ = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  }

  if (!loadAnchors(anchor_file_path)) {
    std::cout << "[ERROR][Detector::Detector]" << std::endl;
    std::cout << "\t loadAnchors failed!" << std::endl;
  }

  if (!loadModel(model_file_path)) {
    std::cout << "[ERROR][Detector::Detector]" << std::endl;
    std::cout << "\t loadModel failed!" << std::endl;
  }
}

const bool Detector::clear() {
  model_.reset();

  return true;
}

const bool Detector::isValid() {
  if (model_ == nullptr) {
    return false;
  }

  if (!anchor_loaded_) {
    return false;
  }

  return true;
}

const bool Detector::loadAnchors(const std::string &anchor_file_path) {
  anchor_loaded_ = false;

  if (!std::filesystem::exists(anchor_file_path)) {
    std::cout << "[ERROR][Detector::loadAnchors]" << std::endl;
    std::cout << "\t anchor file not exist!" << std::endl;
    std::cout << "\t anchor_file_path: " << anchor_file_path << std::endl;

    return false;
  }

  cnpy::NpyArray arr = cnpy::npy_load(anchor_file_path);

  const int anchor_size = arr.shape[0] * arr.shape[1];

  anc_ = torch::from_blob(arr.data<float>(), {long(anchor_size)}, opts_)
             .reshape({1, long(arr.shape[0]), long(arr.shape[1])})
             .detach()
             .clone();

  anchor_loaded_ = true;
  return true;
  ;
}

const bool Detector::loadModel(const std::string &model_file_path) {
  if (!std::filesystem::exists(model_file_path)) {
    std::cout << "[ERROR][Detector::loadModel]" << std::endl;
    std::cout << "\t model file not exist!" << std::endl;
    std::cout << "\t model_file_path: " << model_file_path << std::endl;

    return false;
  }

  model_ = std::make_shared<torch::jit::script::Module>(
      torch::jit::load(model_file_path));

  return true;
}

const bool Detector::detect(const std::vector<float> &points,
                            const int &resolution) {
  const float threshold = std::log(threshold_) - std::log(1.0 - threshold_);

  const float box_size = 1.0f + padding_;

  const torch::Tensor pointsf =
      box_size * make_3d_grid({-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5},
                              {resolution, resolution, resolution});

  const torch::Tensor qry = pointsf.unsqueeze(0).to(opts_);

  torch::Tensor pcd =
      torch::from_blob((void *)(points).data(), {long(points.size())}, opts_)
          .reshape({-1, 3});

  torch::Tensor translate;
  float scale;
  normalizePoints(pcd, translate, scale);

  const torch::Tensor trans_pcd = ((pcd - translate) / scale).unsqueeze(0);

  const torch::Tensor values = eval_points(trans_pcd, qry, anc_);

  return true;
}

const torch::Tensor Detector::eval_points(const torch::Tensor &pcd,
                                          const torch::Tensor &qry,
                                          const torch::Tensor &anc) {
  torch::NoGradGuard no_grad;

  const int n_qry = qry.size(1);

  int chunk_size = int(chunk_size_ * 0.1);

  const int n_chunk = std::ceil(n_qry / chunk_size);

  std::vector<torch::Tensor> ret;

  std::vector<torch::jit::IValue> inputs;
  inputs.resize(3);
  inputs[0] = pcd;
  inputs[2] = anc;

  for (int i = 0; i < n_chunk; ++i) {
    torch::Tensor qry_chunk;
    if (i < n_chunk - 1) {
      qry_chunk = qry.index(
          {Slice(), Slice(chunk_size * i, chunk_size * (i + 1)), Ellipsis});
    } else {
      qry_chunk = qry.index({Slice(), Slice(chunk_size * i, n_qry), Ellipsis});
    }

    inputs[1] = qry_chunk;

    const torch::Tensor occ = model_->forward(inputs).toTensor();

    std::cout << occ.index({0, Slice(None, 10)}) << std::endl;
    std::cout << occ.sizes() << std::endl;

    exit(0);
  }

  return ret[0];
}
