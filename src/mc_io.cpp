#include "mc_io.h"
#include <filesystem>

const bool saveMeshFile(const MC::mcMesh &mesh,
                        const std::string &save_mesh_file_path) {
  open3d::geometry::TriangleMesh o3d_mesh;

  o3d_mesh.vertices_.resize(mesh.vertices.size());
  for (int i = 0; i < mesh.vertices.size(); ++i) {
    o3d_mesh.vertices_[i] = Eigen::Vector3d(
        mesh.vertices[i].x, mesh.vertices[i].y, mesh.vertices[i].z);
  }

  o3d_mesh.vertex_normals_.resize(mesh.normals.size());
  for (int i = 0; i < mesh.normals.size(); ++i) {
    o3d_mesh.vertex_normals_[i] = Eigen::Vector3d(
        mesh.normals[i].x, mesh.normals[i].y, mesh.normals[i].z);
  }

  const int face_num = int(mesh.indices.size() / 3);

  o3d_mesh.triangles_.resize(face_num);
  for (int i = 0; i < face_num; ++i) {
    o3d_mesh.triangles_[i] = Eigen::Vector3i(
        mesh.indices[3 * i], mesh.indices[3 * i + 1], mesh.indices[3 * i + 2]);
  }

  size_t pos = save_mesh_file_path.rfind("/");
  if (pos != std::string::npos) {
    const std::string save_mesh_folder_path =
        save_mesh_file_path.substr(0, pos) + "/";

    if (!std::filesystem::exists(save_mesh_folder_path)) {
      std::filesystem::create_directories(save_mesh_folder_path);
    }
  }

  open3d::io::WriteTriangleMesh(save_mesh_file_path, o3d_mesh);

  return true;
}
