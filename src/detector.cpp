#define MC_IMPLEM_ENABLE
#include "detector.h"
#include "MC.h"
#include "cnpy.h"
#include <filesystem>
#include <memory>
#include <torch/script.h>

Detector::Detector(const std::string &anchor_file_path,
                   const std::string &model_file_path, const bool &use_gpu) {
  if (use_gpu) {
    device_ = torch::kCUDA;
  } else {
    device_ = torch::kCPU;
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

  float *loaded_data = arr.data<float>();

  const int anchor_size = arr.shape[0] * arr.shape[1];

  std::vector<float> anchor_data;
  anchor_data.resize(anchor_size);

  for (int i = 0; i < anchor_size; ++i) {
    anchor_data[i] = loaded_data[i];
  }

  anc_ = torch::from_blob(anchor_data.data(), {long(anchor_data.size())},
                          torch::kFloat32)
             .reshape({1, long(arr.shape[0]), long(arr.shape[1])})
             .to(device_);

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
