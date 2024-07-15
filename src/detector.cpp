#include <memory>
#define MC_IMPLEM_ENABLE
#include "MC.h"
#include "detector.h"
#include <filesystem>
#include <torch/script.h>

Detector::Detector(const std::string &model_file_path) {
  if (!loadModel(model_file_path)) {
    std::cout << "[ERROR][Detector::Detector]" << std::endl;
    std::cout << "\t loadModel failed!" << std::endl;
  }
}

const bool Detector::isValid() { return !model_loaded_; }

const bool Detector::loadModel(const std::string &model_file_path) {
  model_loaded_ = false;

  if (!std::filesystem::exists(model_file_path)) {
    std::cout << "[ERROR][Detector::loadModel]" << std::endl;
    std::cout << "\t model file not exist!" << std::endl;
    std::cout << "\t model_file_path: " << model_file_path << std::endl;

    return false;
  }

  model_ = std::make_shared<torch::jit::script::Module>(
      torch::jit::load(model_file_path));

  model_loaded_ = true;
  return true;
}
