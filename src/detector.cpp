#define MC_IMPLEM_ENABLE
#include "detector.h"
#include "MC.h"

Detector::Detector(const std::string &model_file_path) {
  if (!loadModel(model_file_path)) {
    std::cout << "[ERROR][Detector::Detector]" << std::endl;
    std::cout << "\t loadModel failed!" << std::endl;
  }
}

const bool Detector::isValid() { return !model_loaded_; }

const bool Detector::loadModel(const std::string &model_file_path) {
  model_loaded_ = false;

  model_loaded_ = true;
  return true;
}
