# load all the files under the HRUKBB dire
import os
import sys 
sys.path.append(os.path.join('..'))


import numpy as np
import torch
import torch.nn.functional as F
import data.data_utils as dut
from data_process.dataset_real_scaling import *
from GHD.GHD_cardiac import GHD_Cardiac
from GHD import GHD_config
from ops.medical_related import get_4chamberview_frame
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.io import save_obj

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

datadir = '/media/ssd/fanwen/MultiView/HRUKBB/Dataset'
rootdir = '/media/ssd/fanwen/MultiView/HRUKBB/'
savedir = '/media/ssd/fanwen/MultiView/HRUKBB/GHBMesh'

HRcases = ['HR_ES', 'HR_ED']
casenames = os.listdir(datadir)

root_path = os.path.dirname(os.path.realpath('.'))
base_shape_path = 'canonical_shapes/Standard_LV_2000.obj'
base_shape_path = os.path.join(root_path, base_shape_path)
bi_ventricle_path = 'canonical_shapes/Standard_BiV.obj'
bi_ventricle_path = os.path.join(root_path, bi_ventricle_path)

loss_dict = {'Loss_occupancy':1., 'Loss_normal_consistency':0.01, 'Loss_Laplacian':0.1, 'Loss_thickness':0.02}

# base_shape_path = 'metadata/Standard_LV.obj'
# bi_ventricle_path = 'metadata/Standard_BiV.obj'

cfg = GHD_config(base_shape_path=base_shape_path,
            num_basis=6**2, mix_laplacian_tradeoff={'cotlap':1.0, 'dislap':0.1, 'stdlap':0.1},
            device='cuda:3',
            if_nomalize=True, if_return_scipy=True, 
            bi_ventricle_path=bi_ventricle_path)

paraheart = GHD_Cardiac(cfg) #

for casename in casenames:
    casedir = os.path.join(datadir, casename)
    # sample the HR_ED segs
    for HRcase in HRcases:
        try:
            if not os.path.exists(os.path.join(casedir, (HRcase + '.nii.gz'))):
                continue
            hr_nii_path = os.path.join(casedir, (HRcase + '.nii.gz'))
            output = dut.load_nib_image(hr_nii_path)
            labels = torch.Tensor(output['img']).to(device).unsqueeze(0).unsqueeze(0)
            affines = torch.Tensor(output['affine']).to(device).unsqueeze(0)
            point_list = point_cloud_extractor(labels,  
                                                [0,4,2,1], 
                                                output['window_size'], 
                                                spacing=200, 
                                                coordinate_order = 'zyx')
            coordinate_map_tem = dut.get_coord_map_3d(labels.shape[-3:], 
                                            torch.eye(4).to(device), 
                                            rescaler = 1/100.0)

            Z_rv, Y_rv, X_rv = torch.where(labels[0, 0]==4)
            Z_lv, Y_lv, X_lv = torch.where(labels[0, 0]==2)
            Z_cav, Y_cav, X_cav = torch.where(labels[0, 0]==1)
            Z_bg, Y_bg, X_bg = torch.where(labels[0, 0]==0)
            Z_point, Y_point, X_point = torch.where(labels[0, 0]==3)
            
            b = 0
            Pt_rv = coordinate_map_tem[b, Z_rv, Y_rv, X_rv]
            Pt_lv = coordinate_map_tem[b, Z_lv, Y_lv, X_lv]
            Pt_cav = coordinate_map_tem[b, Z_cav, Y_cav, X_cav]       
            geom_dict = get_4chamberview_frame(Pt_cav, Pt_lv, Pt_rv)   

            # initial global reg.
            initial_orientation = geom_dict['target_affine'].cpu().numpy()
            R = initial_orientation[:3, :3]
            T = initial_orientation[:3, 3]

            paraheart.R = matrix_to_axis_angle(torch.Tensor(R).to(paraheart.device)).view(paraheart.R.shape)
            paraheart.T = torch.from_numpy(T).to(paraheart.device).view(paraheart.T.shape)

            current_mesh = paraheart.rendering()
            current_trimesh_bi = paraheart.rendering_bi_ventricle()

            # cpd rigid alignment using the bi-ventricle mesh
            points_bi = torch.cat(point_list[1:3], dim=0)
            mesh_gt_bi_sample = points_bi.detach().cpu().numpy()[np.random.choice(points_bi.shape[0], 2000, replace=False)]
            paraheart.global_registration_biv(mesh_gt_bi_sample)


            # cpd rigid alignment using the LV mesh
            points_lv = point_list[2]
            mesh_gt_lv_sample = points_lv.detach().cpu().numpy()[np.random.choice(points_lv.shape[0], 2000, replace=False)]
            paraheart.global_registration_lv(mesh_gt_lv_sample)

            # GHB morphing.
            bbox_lv = torch.stack([points_lv.min(dim=0)[0], points_lv.max(dim=0)[0]], dim=0).T
            rescale = 1.1
            bbox_lv_center =  bbox_lv.mean(-1)
            bbox_lv = torch.stack([bbox_lv_center-rescale*(bbox_lv_center-bbox_lv[:,0]), bbox_lv_center+rescale*(bbox_lv[:,1]-bbox_lv_center)], dim=-1)

            points_outoflv = torch.cat([point_list[0], point_list[1]], dim=0)
            points_outoflv_in_bbox = points_outoflv[(points_outoflv[:,0]>bbox_lv[0,0]) & (points_outoflv[:,0]<bbox_lv[0,1]) & (points_outoflv[:,1]>bbox_lv[1,0]) & (points_outoflv[:,1]<bbox_lv[1,1]) & (points_outoflv[:,2]>bbox_lv[2,0]) & (points_outoflv[:,2]<bbox_lv[2,1])]
            points_outoflv_in_bbox = torch.cat([points_outoflv_in_bbox, point_list[-1]], dim=0)

            mesh_after_globalreg = paraheart.rendering()
            # refinement of the LV mesh
            current_mesh, loss_dict = paraheart.morphing2lvtarget(points_lv, 
                                                                    points_outoflv_in_bbox, 
                                                                    target_mesh=None, 
                                                                    loss_dict = loss_dict,
                                                                    lr_start=1e-3, 
                                                                    num_iter=1000, 
                                                                    num_sample=5000, 
                                                                    NP_ratio=1,
                                                                    if_reset=True, 
                                                                    if_fit_R=False, 
                                                                    if_fit_s=True, 
                                                                    if_fit_T=True)
            # save the current_mesh
            save_path = os.path.join(savedir, casename)
            os.makedirs(save_path, exist_ok=True)
            # current_mesh.export(os.path.join(save_path, (HRcase + '_GHD.obj')))
            # 
            verts = current_mesh.verts_list()[0]  # Get vertices as a tensor
            faces = current_mesh.faces_list()[0]  # Get faces as a tensor

            # Convert tensors to float (if necessary)
            verts = verts.float()
            faces = faces.int()

            # Define the output path
            output_path = os.path.join(savedir, casename, (HRcase + '_GHD.obj'))
            # Save as OBJ file
            save_obj(output_path, verts, faces)
            # save the target geom_dict
            np.save(os.path.join(save_path, (HRcase + '_geom_dict.npy')), geom_dict)
        except Exception as e:
            print('Error in case: ', casename, HRcase)
            print(e)
            continue