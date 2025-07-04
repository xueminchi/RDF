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

# load trained model
model_path = os.path.join(CUR_PATH,f'../models_xarm/BP_8.pt')
model = torch.load(model_path, weights_only=False)
print(model[0]['mesh_name'])
# obseve the offset of 
offset = model[0]['offset'].cpu().numpy()
scale = model[0]['scale']
print(f'offset: {offset}, scale: {scale}')
    

mp1 = os.path.join(CUR_PATH, '../collision_avoidance_example/xarm7_urdf/xarm_description/meshes/xarm7/visual/*.stl')
mp2 = os.path.join(CUR_PATH, '../collision_avoidance_example/xarm7_learned_urdf/xarm_description/meshes/xarm7/visual/*.stl')

mesh_list_origin = sorted(glob.glob(mp1))
mesh_list_learned = sorted(glob.glob(mp2))
print(mesh_list_origin)
print(mesh_list_learned)
for i,m1, m2 in zip(range(len(mesh_list_origin)),mesh_list_origin, mesh_list_learned):
    mesh1 = trimesh.load(m1)
    mesh1.visual.face_colors = [255, 0, 0, 255]
    mesh2 = trimesh.load(m2)
    mesh2.visual.face_colors = [0, 255, 0, 255]
    mesh_dict = model[i+1]
    offset = mesh_dict['offset'].cpu().numpy()
    scale = mesh_dict['scale']
    mesh2.vertices = mesh2.vertices*scale + offset
    # mesh2.apply_transform(trans_list[mesh_name].squeeze().cpu().numpy())
    # mesh2.apply_transform(view_mat)
    print(mesh1.vertices.max(axis=0))
    print(mesh2.vertices.max(axis=0))
    print(mesh1.vertices.min(axis=0))
    print(mesh2.vertices.min(axis=0))
    scene = trimesh.Scene()
    scene.add_geometry(mesh1)
    scene.add_geometry(mesh2)
    scene.show()