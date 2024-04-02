import numpy as np
import open3d as o3d


def create_raycast_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def cast_rays(scene, rays):
    rays_o3dt = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    hits = scene.cast_rays(rays_o3dt)
    # dist
    hit_dists = hits["t_hit"].numpy()  # real/inf
    # mask
    hit_geos = hits["geometry_ids"].numpy()
    hit_mask = hit_geos != o3d.t.geometry.RaycastingScene.INVALID_ID
    # hit_ids = np.where(hit_mask)[0]
    hit_dists[~hit_mask] = 1.0
    rdf = np.full_like(hit_dists, 1.0, dtype="float32")
    mask = np.full_like(hit_dists, 0.0, dtype="float32")
    rdf[hit_mask] = hit_dists[hit_mask]
    mask[hit_mask] = hit_mask[hit_mask]
    return rdf, mask
