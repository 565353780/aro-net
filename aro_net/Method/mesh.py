import trimesh
import numpy as np
import open3d as o3d



def load_mesh(file_in):
    mesh = trimesh.load(
        file_in, force="mesh", skip_materials=True, maintain_order=True, process=False
    )
    return mesh
def create_mesh_o3d(v,f):
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v),
        o3d.utility.Vector3iVector(f))
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d



def normalize_mesh(file_in, file_out):
    mesh = load_mesh(file_in)
    assert mesh is not None

    if mesh.extents.min() == 0.0:
        return
    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    scale = 1.0 / mesh.scale
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)

    mesh.export(file_out, include_texture=False)
    return True


def sample_point_cloud(path_mesh, n_pts):
    mesh = load_mesh(path_mesh)
    pts = mesh.sample(n_pts)

    return np.array(pts)
