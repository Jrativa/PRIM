# meshtosdf.py
import trimesh
import numpy as np
from scipy.ndimage import distance_transform_edt

def mesh_to_sdf_npz(obj_path, out_path, resolution=64, padding=5):
    mesh = trimesh.load(obj_path)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Loaded mesh is not a Trimesh object")

    # Normalize: center at origin + scale to fit inside [-0.5, 0.5] (with padding)
    mesh.apply_translation(-mesh.centroid)
    bbox_size = mesh.bounds[1] - mesh.bounds[0]
    scale = 1.0 / (bbox_size.max() * (1 + 2 * padding / resolution))
    mesh.apply_scale(scale)

    # Translate to [0.5, 0.5, 0.5] so it fits inside voxel grid
    mesh.apply_translation([0.5, 0.5, 0.5])

    # Voxelization
    voxels = mesh.voxelized(pitch=1.0 / resolution).fill()
    occupancy = voxels.matrix.astype(np.uint8)

    # Pad grid
    padded = np.pad(occupancy, padding)

    # Distance transforms
    sdf_inside = -distance_transform_edt(padded)
    sdf_outside = distance_transform_edt(1 - padded)
    sdf = sdf_inside
    sdf[padded == 0] = sdf_outside[padded == 0]

    # Crop back to original resolution
    crop = tuple(slice(padding, -padding) for _ in range(3))
    sdf = sdf[crop]

    np.savez_compressed(out_path, sdf=sdf.astype(np.float32))
    print(f"Saved SDF to {out_path} - shape: {sdf.shape}")
