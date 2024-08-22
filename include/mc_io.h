#pragma once

#include <MC.h>
#include <open3d/Open3D.h>

const bool saveMeshFile(const MC::mcMesh &mesh,
                        const std::string &save_mesh_file_path,
                        const std::vector<float> &translate = std::vector<float>{0.0, 0.0, 0.0},
                        const float &scale = 1.0,
                        const int &resolution = 0,
                        const bool &overwrite = false);
