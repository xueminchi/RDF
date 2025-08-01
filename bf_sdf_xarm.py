# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import torch
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import glob
import trimesh
import utils
import mesh_to_sdf
import skimage
# from panda_layer.panda_layer_textured_gripper import PandaLayer
from xarm_layer.xarm_layer import PandaLayer
import argparse

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

class BPSDF():
    def __init__(self, n_func,domain_min,domain_max,robot,device):
        self.n_func = n_func
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.device = device    
        self.robot = robot
        self.model_path = os.path.join(CUR_DIR, 'models_xarm')
        
    def binomial_coefficient(self, n, k):
        return torch.exp(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1))

    def build_bernstein_t(self,t, use_derivative=False):
        # t is normalized to [0,1]
        t =torch.clamp(t, min=1e-4, max=1-1e-4)
        n = self.n_func - 1
        i = torch.arange(self.n_func, device=self.device)
        comb = self.binomial_coefficient(torch.tensor(n, device=self.device), i)
        phi = comb * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** i
        if not use_derivative:
            return phi.float(),None
        else:
            dphi = -comb * (n - i) * (1 - t).unsqueeze(-1) ** (n - i - 1) * t.unsqueeze(-1) ** i + comb * i * (1 - t).unsqueeze(-1) ** (n - i) * t.unsqueeze(-1) ** (i - 1)
            dphi = torch.clamp(dphi, min=-1e4, max=1e4)
            return phi.float(),dphi.float()

    def build_basis_function_from_points(self,p,use_derivative=False):
        N = len(p)
        p = ((p - self.domain_min)/(self.domain_max-self.domain_min)).reshape(-1)
        phi,d_phi = self.build_bernstein_t(p,use_derivative) 
        phi = phi.reshape(N,3,self.n_func)
        phi_x = phi[:,0,:]
        phi_y = phi[:,1,:]
        phi_z = phi[:,2,:]
        phi_xy = torch.einsum("ij,ik->ijk",phi_x,phi_y).view(-1,self.n_func**2)
        phi_xyz = torch.einsum("ij,ik->ijk",phi_xy,phi_z).view(-1,self.n_func**3)
        if use_derivative ==False:
            return phi_xyz,None
        else:
            d_phi = d_phi.reshape(N,3,self.n_func)
            d_phi_x_1D= d_phi[:,0,:]
            d_phi_y_1D = d_phi[:,1,:]
            d_phi_z_1D = d_phi[:,2,:]
            d_phi_x = torch.einsum("ij,ik->ijk",torch.einsum("ij,ik->ijk",d_phi_x_1D,phi_y).view(-1,self.n_func**2),phi_z).view(-1,self.n_func**3)
            d_phi_y = torch.einsum("ij,ik->ijk",torch.einsum("ij,ik->ijk",phi_x,d_phi_y_1D).view(-1,self.n_func**2),phi_z).view(-1,self.n_func**3)
            d_phi_z = torch.einsum("ij,ik->ijk",phi_xy,d_phi_z_1D).view(-1,self.n_func**3)
            d_phi_xyz = torch.cat((d_phi_x.unsqueeze(-1),d_phi_y.unsqueeze(-1),d_phi_z.unsqueeze(-1)),dim=-1)
            return phi_xyz,d_phi_xyz

    def train_bf_sdf(self,epoches=200):
        # represent SDF using basis functions
        # 定义你想要的顺序
        desired_order = [
            "link0",
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
        ]
        # 获取所有文件
        mesh_files = glob.glob(os.path.join(CUR_DIR, "xarm_layer/meshes/*.stl"))
        # 构建 {basename: path} 映射
        mesh_map = {os.path.basename(f).split(".")[0]: f for f in mesh_files}
        # 按理想顺序提取文件路径
        mesh_files_ordered = [mesh_map[name] for name in desired_order if name in mesh_map]
        print("Final mesh file order:")
        for f in mesh_files_ordered:
            print(f)
        
        mesh_dict = {}
        for i,mf in enumerate(mesh_files_ordered):
            mesh_name = mf.split('/')[-1].split('.')[0]
            try:
                mesh = trimesh.load(mf)
            except Exception as e:
                print(f"Failed to load {mf}: {e}")
            offset = mesh.bounding_box.centroid
            scale = np.max(np.linalg.norm(mesh.vertices-offset, axis=1))
            mesh = mesh_to_sdf.scale_to_unit_sphere(mesh)
            mesh_dict[i] = {}
            mesh_dict[i]['mesh_name'] = mesh_name
            # load data
            data = np.load(f'./data/sdf_points/voxel_128_{mesh_name}.npy',allow_pickle=True).item()
            point_near_data = data['near_points']
            sdf_near_data = data['near_sdf']
            point_random_data = data['random_points']
            sdf_random_data = data['random_sdf']
            sdf_random_data[sdf_random_data <-1] = -sdf_random_data[sdf_random_data <-1]
            wb = torch.zeros(self.n_func**3).float().to(self.device)
            B = (torch.eye(self.n_func**3)/1e-4).float().to(self.device)
            # loss_list = []
            for iter in range(epoches):
                choice_near = np.random.choice(len(point_near_data),1024,replace=False)
                p_near,sdf_near = torch.from_numpy(point_near_data[choice_near]).float().to(self.device),torch.from_numpy(sdf_near_data[choice_near]).float().to(self.device)
                choice_random = np.random.choice(len(point_random_data),256,replace=False)
                p_random,sdf_random = torch.from_numpy(point_random_data[choice_random]).float().to(self.device),torch.from_numpy(sdf_random_data[choice_random]).float().to(self.device)
                p = torch.cat([p_near,p_random],dim=0)
                sdf = torch.cat([sdf_near,sdf_random],dim=0)
                phi_xyz, _ = self.build_basis_function_from_points(p.float().to(self.device),use_derivative=False)

                K = torch.matmul(B,phi_xyz.T).matmul(torch.linalg.inv((torch.eye(len(p)).float().to(self.device)+torch.matmul(torch.matmul(phi_xyz,B),phi_xyz.T))))
                B -= torch.matmul(K,phi_xyz).matmul(B)
                delta_wb = torch.matmul(K,(sdf - torch.matmul(phi_xyz,wb)).squeeze())
                # loss = torch.nn.functional.mse_loss(torch.matmul(phi_xyz,wb).squeeze(), sdf, reduction='mean').item()
                # loss_list.append(loss)
                wb += delta_wb

            print(f'mesh name {mesh_name} finished!')
            mesh_dict[i] ={
                'mesh_name':     mesh_name,
                'weights':  wb,
                'offset':   torch.from_numpy(offset),
                'scale':      scale,  

            }
        if os.path.exists(self.model_path) is False:
            os.mkdir(self.model_path)
        torch.save(mesh_dict,f'{self.model_path}/BP_{self.n_func}.pt') # save the robot sdf model
        print(f'{self.model_path}/BP_{self.n_func}.pt model saved!')

    def sdf_to_mesh(self, model, nbData,use_derivative=False):
        verts_list, faces_list, mesh_name_list = [], [], []
        for i, k in enumerate(model.keys()):
            mesh_dict = model[k]
            mesh_name = mesh_dict['mesh_name']
            print(f'{mesh_name}')
            mesh_name_list.append(mesh_name)
            weights = mesh_dict['weights'].to(self.device)

            domain = torch.linspace(self.domain_min,self.domain_max,nbData).to(self.device)
            grid_x, grid_y, grid_z= torch.meshgrid(domain,domain,domain)
            grid_x, grid_y, grid_z = grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1,1)
            p = torch.cat([grid_x, grid_y, grid_z],dim=1).float().to(self.device)   

            # split data to deal with memory issues
            p_split = torch.split(p, 10000, dim=0)
            d =[]
            for p_s in p_split:
                phi_p,d_phi_p = self.build_basis_function_from_points(p_s,use_derivative)
                d_s = torch.matmul(phi_p,weights)
                d.append(d_s)
            d = torch.cat(d,dim=0)

            verts, faces, normals, values = skimage.measure.marching_cubes(
                d.view(nbData,nbData,nbData).detach().cpu().numpy(), level=0.0, spacing=np.array([(self.domain_max-self.domain_min)/nbData] * 3)
            )
            verts = verts - [1,1,1]
            verts_list.append(verts)
            faces_list.append(faces)
        return verts_list, faces_list,mesh_name_list

    def create_surface_mesh(self,model, nbData,vis =False, save_mesh_name=None):
        verts_list, faces_list,mesh_name_list = self.sdf_to_mesh(model, nbData)
        for verts, faces,mesh_name in zip(verts_list, faces_list,mesh_name_list):
            rec_mesh = trimesh.Trimesh(verts,faces)
            if vis:
                rec_mesh.show()
            if save_mesh_name != None:
                save_path = os.path.join(CUR_DIR,"output_meshes")
                if os.path.exists(save_path) is False:
                    os.mkdir(save_path)
                trimesh.exchange.export.export_mesh(rec_mesh, os.path.join(save_path,f"{save_mesh_name}_{mesh_name}.stl"))

    def get_whole_body_sdf_batch(self,x,pose,theta,model,use_derivative = True, used_links = [0,1,2,3,4,5,6,7],return_index=False):

        B = len(theta)
        N = len(x)
        K = len(used_links)
        offset = torch.cat([model[i]['offset'].unsqueeze(0) for i in used_links],dim=0).to(self.device)
        offset = offset.unsqueeze(0).expand(B,K,3).reshape(B*K,3).float()
        scale = torch.tensor([model[i]['scale'] for i in used_links],device=self.device)
        scale = scale.unsqueeze(0).expand(B,K).reshape(B*K).float()
        trans_list = self.robot.forward_kinematics(theta)
        trans_list = list(trans_list.values())
        trans_list = trans_list[1:]
        fk_trans = torch.cat([t.unsqueeze(1) for t in trans_list],dim=1)[:,used_links,:,:].reshape(-1,4,4) # B,K,4,4
        x_robot_frame_batch = utils.transform_points(x.float(),torch.linalg.inv(fk_trans).float(),device=self.device) # B*K,N,3
        x_robot_frame_batch_scaled = x_robot_frame_batch - offset.unsqueeze(1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled/scale.unsqueeze(-1).unsqueeze(-1) #B*K,N,3

        x_bounded = torch.where(x_robot_frame_batch_scaled>1.0-1e-2,1.0-1e-2,x_robot_frame_batch_scaled)
        x_bounded = torch.where(x_bounded<-1.0+1e-2,-1.0+1e-2,x_bounded)
        res_x = x_robot_frame_batch_scaled - x_bounded

        if not use_derivative:
            phi,_ = self.build_basis_function_from_points(x_bounded.reshape(B*K*N,3), use_derivative=False)
            phi = phi.reshape(B,K,N,-1).transpose(0,1).reshape(K,B*N,-1) # K,B*N,-1
            weights_near = torch.cat([model[i]['weights'].unsqueeze(0) for i in used_links],dim=0).to(self.device)
            # sdf
            sdf = torch.einsum('ijk,ik->ij',phi,weights_near).reshape(K,B,N).transpose(0,1).reshape(B*K,N) # B,K,N
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B,K,N)
            sdf = sdf*scale.reshape(B,K).unsqueeze(-1)
            sdf_value, idx = sdf.min(dim=1)
            if return_index:
                return sdf_value, None, idx
            return sdf_value, None
        else:   
            phi,dphi = self.build_basis_function_from_points(x_bounded.reshape(B*K*N,3), use_derivative=True)
            phi_cat = torch.cat([phi.unsqueeze(-1),dphi],dim=-1)
            phi_cat = phi_cat.reshape(B,K,N,-1,4).transpose(0,1).reshape(K,B*N,-1,4) # K,B*N,-1,4

            weights_near = torch.cat([model[i]['weights'].unsqueeze(0) for i in used_links],dim=0).to(self.device)

            output = torch.einsum('ijkl,ik->ijl',phi_cat,weights_near).reshape(K,B,N,4).transpose(0,1).reshape(B*K,N,4)
            sdf = output[:,:,0]
            gradient = output[:,:,1:]
            # sdf
            sdf = sdf + res_x.norm(dim=-1)
            sdf = sdf.reshape(B,K,N)
            sdf = sdf*(scale.reshape(B,K).unsqueeze(-1))
            sdf_value, idx = sdf.min(dim=1)
            # derivative
            gradient = res_x + torch.nn.functional.normalize(gradient,dim=-1)
            gradient = torch.nn.functional.normalize(gradient,dim=-1).float()
            # gradient = gradient.reshape(B,K,N,3)
            fk_rotation = fk_trans[:,:3,:3]
            gradient_base_frame = torch.einsum('ijk,ikl->ijl',fk_rotation,gradient.transpose(1,2)).transpose(1,2).reshape(B,K,N,3)
            # norm_gradient_base_frame = torch.linalg.norm(gradient_base_frame,dim=-1)

            # exit()
            # print(norm_gradient_base_frame)

            idx_grad = idx.unsqueeze(1).unsqueeze(-1).expand(B,K,N,3)
            gradient_value = torch.gather(gradient_base_frame,1,idx_grad)[:,0,:,:]
            # gradient_value = None
            if return_index:
                return sdf_value, gradient_value, idx
            return sdf_value, gradient_value

    def get_whole_body_sdf_with_joints_grad_batch(self,x,pose,theta,model,used_links = [0,1,2,3,4,5,6,7]):

        delta = 0.001
        B = theta.shape[0]
        theta = theta.unsqueeze(1)
        d_theta = (theta.expand(B,7,7)+ torch.eye(7,device=self.device).unsqueeze(0).expand(B,7,7)*delta).reshape(B,-1,7)
        theta = torch.cat([theta,d_theta],dim=1).reshape(B*8,7)
        pose = pose.unsqueeze(1).expand(B,8,4,4).reshape(B*8,4,4)
        sdf,_ = self.get_whole_body_sdf_batch(x,pose,theta,model,use_derivative = False, used_links = used_links)
        sdf = sdf.reshape(B,8,-1)
        d_sdf = (sdf[:,1:,:]-sdf[:,:1,:])/delta
        return sdf[:,0,:],d_sdf.transpose(1,2)

    def get_whole_body_normal_with_joints_grad_batch(self,x,pose,theta,model,used_links = [0,1,2,3,4,5,6,7]):
        delta = 0.001
        B = theta.shape[0]
        theta = theta.unsqueeze(1)
        d_theta = (theta.expand(B,7,7)+ torch.eye(7,device=self.device).unsqueeze(0).expand(B,7,7)*delta).reshape(B,-1,7)
        theta = torch.cat([theta,d_theta],dim=1).reshape(B*8,7)
        pose = pose.unsqueeze(1).expand(B,8,4,4).reshape(B*8,4,4)
        sdf, normal = self.get_whole_body_sdf_batch(x,pose,theta,model,use_derivative = True, used_links = used_links)
        normal = normal.reshape(B,8,-1,3).transpose(1,2)
        return normal # normal size: (B,N,8,3) normal[:,:,0,:] origin normal vector normal[:,:,1:,:] derivatives with respect to joints


if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--domain_max', default=1.0, type=float)
    parser.add_argument('--domain_min', default=-1.0, type=float)
    parser.add_argument('--n_func', default=16, type=int)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    panda = PandaLayer(args.device)
    bp_sdf = BPSDF(args.n_func,args.domain_min,args.domain_max,panda,args.device)
    
    # args.train = True

    # #  train Bernstein Polynomial model   
    if args.train:
        bp_sdf.train_bf_sdf()


    # load trained model
    model_path = os.path.join(CUR_DIR,f'models_xarm/BP_{args.n_func}.pt')
    model = torch.load(model_path, weights_only=False)
    
    # # visualize the Bernstein Polynomial model for each robot link
    # bp_sdf.create_surface_mesh(model,nbData=128,vis=True,save_mesh_name=f'BP_{args.n_func}')

    # visualize the Bernstein Polynomial model for the whole body
    theta = torch.tensor([-0.0, -0.0,  0.0, -0.0, -0.0,  0.0, -0.0]).float().to(args.device).reshape(-1,7)
    pose = torch.from_numpy(np.identity(4)).to(args.device).reshape(-1, 4, 4).expand(len(theta),4,4).float()

    trans_list = panda.forward_kinematics(theta)
    print(trans_list.keys())


    trans_list = list(trans_list.values())
    trans_list = trans_list[1:]
    utils.visualize_reconstructed_whole_body(model, trans_list, tag=f'BP_{args.n_func}')

    # run RDF 
    # x = torch.rand(128,3).to(args.device)*2.0 - 1.0
    x = utils.get_surface_points(model, trans_list, tag=f'BP_{args.n_func}')
    x = torch.from_numpy(x).to(args.device).float()
    x = x[torch.randperm(x.shape[0])[:1000]]
    utils.visualize_reconstructed_whole_body(model, trans_list, tag=f'BP_{args.n_func}', surface_points=x)

    pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(args.device).expand(len(theta),4,4).float()
    sdf,gradient = bp_sdf.get_whole_body_sdf_batch(x,pose,theta,model,use_derivative=True)
    print('sdf:',sdf.max(),sdf.min(),'gradient:',gradient.shape)
    sdf,joint_grad = bp_sdf.get_whole_body_sdf_with_joints_grad_batch(x,pose,theta,model)
    print('sdf:',sdf.shape,'joint gradient:',joint_grad.shape)





