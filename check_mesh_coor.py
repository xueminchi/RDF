import os
import numpy as np
import trimesh
import glob

def get_mesh_bounds(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')
    bounds = mesh.bounds
    min_xyz, max_xyz = bounds
    return min_xyz, max_xyz

def check_mesh_center(stl_path, method='centroid', tol=1e-6):
    """
    检查STL文件的中心是否在(0,0,0)

    参数:
        stl_path: str, STL 文件路径
        method: str, 'centroid'（质心） 或 'bbox'（包围盒中心）
        tol: float, 误差容忍阈值

    返回:
        bool, 是否居中
        numpy.ndarray, 实际的中心坐标
    """
    mesh = trimesh.load_mesh(stl_path)

    if method == 'centroid':
        center = mesh.centroid
    elif method == 'bbox':
        min_corner, max_corner = mesh.bounds
        center = (min_corner + max_corner) / 2
    else:
        raise ValueError("method 只能是 'centroid' 或 'bbox'")

    is_centered = np.allclose(center, [0, 0, 0], atol=tol)

    return is_centered, center


def main():
    # mesh_dir = os.path.join(os.path.dirname(__file__), "output_meshes")
    mesh_dir = os.path.join(os.path.dirname(__file__), "collision_avoidance_example/xarm7_learned_urdf/xarm_description/meshes/xarm7/visual")
    mesh_files = glob.glob(os.path.join(mesh_dir, "*.stl"))
    mesh_files.sort()

    if not mesh_files:
        print(f"❌ 没有找到 STL 文件在 {mesh_dir}")
        return

    for mesh_path in mesh_files:
        min_xyz, max_xyz = get_mesh_bounds(mesh_path)
        is_centered, center = check_mesh_center(mesh_path, method='centroid')

        print(f"📄 {os.path.basename(mesh_path)}")
        print(f"   X: {min_xyz[0]:.6f} ~ {max_xyz[0]:.6f}")
        print(f"   Y: {min_xyz[1]:.6f} ~ {max_xyz[1]:.6f}")
        print(f"   Z: {min_xyz[2]:.6f} ~ {max_xyz[2]:.6f}")
        print(f"   ➡️  Centroid: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
        print(f"   🎯 Is centered (centroid): {'✅' if is_centered else '❌'}")
        print("-" * 50)

if __name__ == "__main__":
    main()
