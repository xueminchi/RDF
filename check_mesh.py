import trimesh
import numpy as np
import glob
import os

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


# 示例用法
# 改成你自己的目录
folder = "/home/ps/py_project/RDF/collision_avoidance_example/xarm7_learned_urdf/xarm_description/meshes/xarm7/visual/"
# folder = "/home/ps/py_project/RDF/collision_avoidance_example/xarm7_urdf/xarm_description/meshes/xarm7/visual"
# folder = "/home/ps/py_project/RDF/panda_layer//meshes/visual/"
stl_files = glob.glob(os.path.join(folder, "*.stl"))

for stl_file in stl_files:
    ok, center = check_mesh_center(stl_file, method='centroid')
    if ok:
        print(f"✅ {os.path.basename(stl_file)} 已居中于 (0,0,0)")
    else:
        print(f"⚠️  {os.path.basename(stl_file)} 未居中，实际中心: {center}")