# -----------------------------------------------------------------------------
#SPDX-License-Identifier: MIT
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

# panda layer implementation using pytorch kinematics
import torch
import trimesh
import glob
import os
import numpy as np
import pytorch_kinematics as pk
import copy
CUR_PATH = os.path.dirname(os.path.realpath(__file__))

def save_to_mesh(vertices, faces, output_mesh_path=None):
    assert output_mesh_path is not None
    with open(output_mesh_path, 'w') as fp:
        for vert in vertices:
            fp.write('v %f %f %f\n' % (vert[0], vert[1], vert[2]))
        for face in faces+1:
            fp.write('f %d %d %d\n' % (face[0], face[1], face[2]))
    print('Output mesh save to: ', os.path.abspath(output_mesh_path))

# mp = os.path.join(CUR_PATH, '../collision_avoidance_example/xarm7_gripper_urdf/xarm_description/meshes/xarm7/visual/*.stl')

mp = [
    os.path.join(CUR_PATH, '../collision_avoidance_example/xarm7_gripper_urdf/xarm_description/meshes/xarm7/visual/*.stl'),
    os.path.join(CUR_PATH, '../collision_avoidance_example/xarm7_gripper_urdf/xarm_gripper/meshes/*.STL')  # 注意大写 .STL
]

class PandaLayer(torch.nn.Module):
    def __init__(self, device='cpu', mesh_path = mp):
        super().__init__()
        dir_path = os.path.split(os.path.abspath(__file__))[0]
        self.device = device
        self.urdf_path = os.path.join(dir_path, '../collision_avoidance_example/xarm7_gripper_urdf/xarm7_with_gripper.urdf')
        self.mesh_path_list = mesh_path
        print('', self.mesh_path_list)
        info_chain = pk.build_chain_from_urdf(open(self.urdf_path, mode="rb").read())
        print('The kinematics chain of the arm is ', info_chain)
        self.chain = pk.build_serial_chain_from_urdf(open(self.urdf_path).read().encode(),"xarm_gripper_base_link").to(dtype = torch.float32,device = self.device)
        joint_lim = torch.tensor(self.chain.get_joint_limits())
        self.theta_min = joint_lim[:,0].to(self.device)
        self.theta_max = joint_lim[:,1].to(self.device)

        self.theta_mid = (self.theta_min + self.theta_max) / 2.0
        self.theta_min_soft = (self.theta_min-self.theta_mid)*0.8 + self.theta_mid
        self.theta_max_soft = (self.theta_max-self.theta_mid)*0.8 + self.theta_mid
        self.dof = len(self.theta_min)
        self.meshes = self.load_meshes()


    def load_meshes(self):
        mesh_files = []
        for path in self.mesh_path_list:  # 新增 mesh_path_list
            mesh_files.extend(glob.glob(path))
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]

        meshes = {}
        for mesh_file in mesh_files:
            name = os.path.splitext(os.path.basename(mesh_file))[0].lower()
            try:
                mesh = trimesh.load(mesh_file, force='mesh')
                if isinstance(mesh, trimesh.Trimesh):
                    meshes[name] = mesh
            except Exception as e:
                print(f"Failed loading {mesh_file}: {e}")
        return meshes


    def forward_kinematics(self, theta):
        ret = self.chain.forward_kinematics(theta, end_only=False)
        transformations = {}
        for k in ret.keys():
            trans_mat = ret[k].get_matrix()
            transformations[k] = trans_mat
        return transformations



    def theta2mesh(self, theta):
        trans = self.forward_kinematics(theta)
        # trans = self.add_gripper_transforms(trans)

        robot_mesh = []
        # print('trans', trans.keys())
        # print('meshes', self.meshes.keys())

        # 确认meshes非空
        for k in self.meshes.keys():
            mesh = self.meshes[k]
            print(f"{k}: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")

        for k in self.meshes.keys():
            mesh = copy.deepcopy(self.meshes[k])
            vertices = torch.from_numpy(mesh.vertices).to(self.device).float()
            vertices = torch.cat([vertices, torch.ones([vertices.shape[0], 1], device=self.device)], dim=-1).t()

            if k in trans:
                transform = trans[k].squeeze()
            elif k == 'link_base':
                print(f"Manually adding transform for '{k}'")
                transform = torch.eye(4, device=self.device)  # 恒等矩阵，表示世界坐标原点
            else:
                print(f"Skipping mesh '{k}' because no FK result was found.")
                continue

            transformed_vertices = torch.matmul(transform, vertices).t()[:, :3].detach().cpu().numpy()
            mesh.vertices = transformed_vertices
            robot_mesh.append(mesh)

        return robot_mesh



if __name__ == "__main__":
    device = 'cuda'
    panda = PandaLayer(device).to(device)
    scene = trimesh.Scene()
    # theta = torch.tensor([-1.7370, -0.0455,  0.1577, -2.8271, -2.0578,  1.8342, -0.1893]).float().to(device).reshape(-1,7)
    theta = torch.tensor([-0.0, -0.0,  0.0, -0.0, -0.0,  0.0, -0.0]).float().to(device).reshape(-1,7)
    trans = panda.forward_kinematics(theta)
    robot_mesh = panda.theta2mesh(theta)
    scene.add_geometry(robot_mesh)
    scene.show()

